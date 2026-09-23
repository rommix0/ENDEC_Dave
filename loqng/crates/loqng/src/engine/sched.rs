//! Cooperative threads for the translated engine.
//!
//! The engine creates one worker, "Sintesi", and hands synthesis to it:
//! `ttsRead` queues the text and returns, so without a worker nothing is ever
//! rendered. That much `loqrs` already established.
//!
//! A translated function is ordinary Rust, deep in a Rust call stack, so it
//! cannot be suspended mid-way the way an interpreter's guest can. Each guest
//! thread therefore gets a real OS thread — but **exactly one runs at a
//! time**, passing a baton. Switches happen only where the engine itself
//! blocks:
//!
//! ```text
//! pthread_cond_wait   give the baton up, wait to be signalled and handed it
//! pthread_join        give it up until the other thread has finished
//! Runtime::pump       the caller lending the baton so the worker can drain
//! ```
//!
//! Because those points are fixed and nothing preempts, the interleaving is
//! deterministic and so is the output — which is the whole requirement.
//!
//! Mutexes need no state for the same reason: with no preemption, a lock can
//! never be contended.
//!
//! A scheduler belongs to one [`Runtime`](super::Runtime) and not to the
//! process. Sharing one across engines let a worker parked from a previous
//! utterance wake up holding pointers into memory that had since been freed.

use std::sync::{Arc, Condvar, Mutex};

pub const MAIN: usize = 0;

struct State {
    /// Whose turn it is to run.
    turn: usize,
    /// The condition variable each thread is blocked on, or 0 if runnable.
    waiting: Vec<u32>,
    done: Vec<bool>,
    /// Set when the engine is going away; parked threads return instead of
    /// touching memory that is about to be dropped.
    shutdown: bool,
}

pub struct Sched {
    st: Mutex<State>,
    cv: Condvar,
}

thread_local! {
    static TID: std::cell::Cell<usize> = const { std::cell::Cell::new(MAIN) };
}

pub fn tid() -> usize {
    TID.with(|t| t.get())
}

pub fn set_tid(k: usize) {
    TID.with(|t| t.set(k));
}

impl Sched {
    pub fn new() -> Arc<Sched> {
        Arc::new(Sched {
            st: Mutex::new(State {
                turn: MAIN,
                waiting: vec![0],
                done: vec![false],
                shutdown: false,
            }),
            cv: Condvar::new(),
        })
    }

    /// Register a new guest thread and return its index.
    pub fn enrol(&self) -> usize {
        let mut st = self.st.lock().unwrap();
        st.waiting.push(0);
        st.done.push(false);
        st.waiting.len() - 1
    }

    fn runnable(st: &State, me: usize) -> Option<usize> {
        (0..st.waiting.len()).find(|&k| k != me && !st.done[k] && st.waiting[k] == 0)
    }

    /// Give the baton to another runnable thread and block until it returns.
    ///
    /// `false` means nothing else could run, so the caller keeps it — in the
    /// guest that is a deadlock, and saying so beats hanging.
    pub fn hand_off(&self, me: usize) -> bool {
        let mut st = self.st.lock().unwrap();
        if st.shutdown {
            return false;
        }
        let Some(next) = Self::runnable(&st, me) else {
            st.waiting[me] = 0;
            return false;
        };
        st.turn = next;
        self.cv.notify_all();
        while st.turn != me && !st.shutdown {
            st = self.cv.wait(st).unwrap();
        }
        !st.shutdown
    }

    /// Block until this thread holds the baton. `false` means give up.
    pub fn wait_turn(&self, me: usize) -> bool {
        let mut st = self.st.lock().unwrap();
        while st.turn != me && !st.shutdown {
            st = self.cv.wait(st).unwrap();
        }
        !st.shutdown
    }

    /// `pthread_cond_wait`: block on `cond` until someone signals it.
    pub fn cond_wait(&self, me: usize, cond: u32) -> bool {
        {
            let mut st = self.st.lock().unwrap();
            st.waiting[me] = cond.max(1);
        }
        self.hand_off(me)
    }

    /// `pthread_cond_signal` / `broadcast`: its waiters become runnable.
    pub fn cond_signal(&self, cond: u32) {
        let mut st = self.st.lock().unwrap();
        let want = cond.max(1);
        for k in 0..st.waiting.len() {
            if st.waiting[k] == want {
                st.waiting[k] = 0;
            }
        }
    }

    pub fn finished(&self, k: usize) -> bool {
        let st = self.st.lock().unwrap();
        st.done.get(k).copied().unwrap_or(true)
    }

    /// Mark this thread finished and pass the baton on.
    pub fn finish(&self, me: usize) {
        let mut st = self.st.lock().unwrap();
        st.done[me] = true;
        st.turn = Self::runnable(&st, me).unwrap_or(MAIN);
        self.cv.notify_all();
    }

    /// Release every parked thread so they can exit.
    pub fn shutdown(&self) {
        let mut st = self.st.lock().unwrap();
        st.shutdown = true;
        self.cv.notify_all();
    }
}
