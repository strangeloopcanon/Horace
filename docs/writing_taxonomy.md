# Writing Style Taxonomy

A reference taxonomy of observable, classifiable style signals drawn from rhetoric, linguistics, editing, and stylometry. Used to guide feature expansion in the Horace scoring system.

## How this document is used

The Horace v6 scorer starts with ~45 features (cadence backbone from causal LM analysis + cheap taxonomy-derived features). This taxonomy serves as the menu for future expansion: after training, error analysis on misranked pairs identifies which categories the model is blind to, and we add features from those categories.

Features currently captured by the scorer are marked with a check. Categories with no checks are candidates for future feature engineering.

---

## 1. Cadence and rhythm

Micro-to-macro "time feel" in prose.

- [x] **Sentence-length profile** — mean, variance, deliberate alternation (`sent_words_mean`, `sent_words_cv`, `sent_len_cv`)
- [x] **Surprisal rhythm** — spike rate, inter-peak intervals, autocorrelation, periodicity (`high_surprise_rate_per_100`, `ipi_mean`, `ipi_cv`, `acf1`, `acf2`, `peak_period`)
- [x] **Run structure** — lengths of low/high surprise runs (`run_low_mean_len`, `run_high_mean_len`)
- [x] **Burstiness** — variation of surprisal across sentences and paragraphs (`sent_burst_cv`, `para_burst_cv`)
- [x] **Punctuation timing** — comma, semicolon, dash, colon density (`comma_per_100w`, `semicolon_per_100w`, `dash_per_100w`, `colon_per_100w`)
- [x] **Function words as metronome** — function word rate (`function_word_rate`)
- [ ] **Stress and beat** — syllable stress clustering, iambic drift
- [ ] **Parallelism and balance** — isocolon, tricolon, antithesis detection
- [ ] **Repetition devices** — anaphora, epistrophe, anadiplosis, polysyndeton, asyndeton
- [ ] **Euphony / cacophony** — alliteration, assonance, consonance
- [ ] **Paragraph rhythm** — placement of "turn" and "drop" sentences

## 2. Word usage and diction

- [x] **Vocabulary richness** — type-token ratio, hapax ratio (`word_ttr`, `word_hapax_ratio`)
- [x] **Word complexity** — mean word length, syllables per word (`word_len_mean`, `syllables_per_word_mean`)
- [ ] **Register** — formal/informal, colloquial markers
- [ ] **Frequency band selection** — preference for high/low frequency words
- [ ] **Concreteness and imagery** — concrete vs abstract nouns
- [ ] **Specificity** — "vehicle" vs "2012 Corolla"
- [ ] **Emotional valence** — sentiment volatility
- [ ] **Hedges and certainty** — "maybe" vs "clearly"
- [ ] **Intensifiers** — "very", "really", "quite" frequency
- [ ] **Jargon density** — technical terms per 1k words

## 3. Lexicon structure

- [x] **Lexical diversity** — TTR, hapax ratio
- [x] **Repetition patterns** — adjacent word, bigram, trigram repeat rates (`adjacent_word_repeat_rate`, `bigram_repeat_rate`, `trigram_repeat_rate`)
- [ ] **Morphological preferences** — Latinate nominalizations vs Germanic verbs
- [ ] **Collocational style** — signature bigrams/trigrams (Burrows' Delta)
- [ ] **Semantic fields** — recurrent metaphor domains
- [ ] **Synonym strategy** — repetition for cohesion vs synonym churn

## 4. Syntax and sentence architecture

- [x] **Question and exclamation rates** (`question_rate`, `exclamation_rate`)
- [x] **Single-sentence paragraph rate** (`one_sent_para_rate`)
- [ ] **Parataxis vs hypotaxis** — coordination vs subordination
- [ ] **Embedding depth** — nested clause depth
- [ ] **Fragmentation** — sentence fragment rate
- [ ] **Inversion and marked word order**
- [ ] **Active vs passive voice**
- [ ] **Nominalization vs verb-driven prose**
- [ ] **Modifier strategy** — pre- vs post-modification

## 5. Grammar choices

- [ ] **Tense and aspect profile**
- [ ] **Mood and modality** — modal verb distribution
- [ ] **Person and deixis** — 1st/2nd/3rd person distribution
- [ ] **Pronoun economy**
- [ ] **Negation style** — litotes, double negation

## 6. Punctuation and orthography

- [x] **Punctuation variety** — distinct punctuation types used (`punct_variety_per_1000_chars`)
- [x] **Per-type rates** — comma, semicolon, dash, colon, parenthetical (`comma_per_100w`, `semicolon_per_100w`, `dash_per_100w`, `colon_per_100w`, `parenthetical_rate`)
- [ ] **Comma philosophy** — minimalist vs rhetorical; Oxford comma
- [ ] **Dash style** — em dash vs en dash; interruption vs parenthetical
- [ ] **Ellipsis and trailing thought**
- [ ] **Quotation behaviour**
- [ ] **Capitalization style**

## 7. Rhetorical figures and devices

- [ ] **Tropes** — metaphor, simile, metonymy, irony, hyperbole, litotes
- [ ] **Schemes** — chiasmus, zeugma, aposiopesis
- [ ] **Argumentative rhetoric** — concession + pivot, rhetorical questions, prolepsis
- [ ] **Humor mechanics** — incongruity, deadpan, callback, bathos

## 8. Cohesion and coherence

- [x] **Narrative structure** — cohesion delta (original vs shuffled) (`cohesion_delta`)
- [x] **Discourse markers** — "however", "therefore", etc. per word (`discourse_marker_rate`)
- [ ] **Referential cohesion** — pronoun chaining, definite descriptions
- [ ] **Lexical cohesion** — repetition, synonymy, hyponymy
- [ ] **Topic management** — theme-rheme progression
- [ ] **Signposting density** — "first/second/third", "in summary"

## 9. Information density and readability

- [x] **Readability proxies** — word length, syllable count, sentence length (captured indirectly)
- [ ] **Idea density** — propositions per clause
- [ ] **Redundancy vs terseness** — restatement rate
- [ ] **Cognitive load management** — garden-path sentence rate

## 10. Tone, stance, and voice

- [ ] **Stance markers** — certainty, doubt, evaluation
- [ ] **Politeness strategy**
- [ ] **Distance** — impersonal constructions
- [ ] **Authority signaling** — definitions, classifications, citations
- [ ] **Cynicism vs earnestness** — irony rate, scare quotes

## 11. Figurative and sensory texture

- [ ] **Metaphor families** — war, journey, machine, ecology
- [ ] **Sensory distribution** — visual vs tactile vs auditory
- [ ] **Symbolic motifs** — recurring objects/colors/weather
- [ ] **Synesthesia** — cross-modal mappings

## 12. Narrative and temporal control

- [ ] **Point of view** — 1st/2nd/3rd, free indirect discourse
- [ ] **Time handling** — linear vs braided; flashback/flashforward
- [ ] **Scene vs summary ratio**
- [ ] **Pacing** — sentence length controlling speed
- [ ] **Openings and endings** — cold open, circular, thesis-first

## 13. Dialogue and voice differentiation

- [ ] **Dialogue realism** — fragments, interruptions, fillers
- [ ] **Tag strategy** — "said" economy vs ornate tags
- [ ] **Idiolect differentiation** — per-character lexicon and cadence

## 14. Paragraphing, layout, and visual rhetoric

- [x] **Paragraph length variation** (`para_len_cv`)
- [x] **Single-sentence paragraphs** (`one_sent_para_rate`)
- [ ] **Typographic emphasis** — italics, quotes, caps
- [ ] **Heading style** — descriptive vs playful
- [ ] **List architecture** — parallel structure discipline

## 15. Genre conventions

- [ ] **Academic** — citations, hedging, nominalizations, passive
- [ ] **Journalistic** — nut graf, attribution verbs, inverted pyramid
- [ ] **Legal** — definitions, shall/may, enumerations
- [ ] **Technical** — imperatives, stepwise procedures
- [ ] **Literary** — deviation, compression, ambiguity tolerance

## 16-20. Advanced categories

These categories (intertextuality, revision fingerprints, stylometry atoms, ethical posture, negative space) are primarily useful for deep author analysis and forensic stylometry rather than general writing quality scoring. They remain available for future specialized features.

---

## Using this taxonomy for feature expansion

1. Train the v6 scorer on current ~45 features
2. Run error analysis on misranked pairs
3. Identify which taxonomy categories the errors cluster in
4. Add computable features from those categories
5. Retrain and evaluate
