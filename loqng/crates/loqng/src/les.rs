//! `Les` — lessicalizzatore: tokenise, expand, look up. **Phase 5.**
//!
//! 39,684 bytes here, plus `ELQ-numbers` (7,588) and `ELQ-automata` (9,676),
//! plus `LesElaboraParola_English` (3,524 B) and `LesIniNumeri_US` in the
//! language module.
//!
//! These decompose into independently testable pieces, which is what makes
//! this stage tractable:
//!
//! ```text
//! LesLoadTextInChunk   0x57924    LesSplitChunk       0x593d8
//! LesNumero2Testo      0x59fa4    LesParola2Testo     0x5a084
//! LesSpeciale2Testo    0x5a12c    LesIsSiglaPuntata   0x53148
//! LesNormalizzaAccenti 0x53434    LesConvertInternet  0x59158
//! LesFindInLexicon     0x617c0    LesLocateEntry      0x61a5c
//! LesGradiAccentualiStandard  0x5a1d4
//! LesConfiniProsodiciStandard 0x5a7c8
//! ```
//!
//! Start with the number converters. They are pure text-in/text-out functions
//! and they cover what actually breaks in production:
//!
//! ```text
//! ELQNumConvertInteger     0x8ef1c    ELQNumConvertCurrency    0x8f0c4
//! ELQNumConvertRealNumber  0x8ea44    ELQNumConvertPhoneNumber 0x8ecec
//! ELQNumConvertBigInteger  0x8e9c8    ELQNumExtractDate        0x8f560
//! ELQNumArabic2Roman       0x8ff04    ELQNumIsTime             0x8e7a4
//! ELQNumIsCurrency         0x9044c    ELQNumAddSeparator       0x8f4a4
//! ```
//!
//! This is also where the known `LoqEnglish6.8` currency-doubling bug lives
//! ("247 pounds" -> "pounds pounds"). Reproduce the 6.9 behaviour, not 6.8's.
//!
//! Gate: `PlainLesOut` matches byte-for-byte across the corpus.
