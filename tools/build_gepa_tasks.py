#!/usr/bin/env python3
"""Construct a curated set of GEPA training tasks from classic authors."""

import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional, Tuple

DOCS_PATH = Path("data/analysis/gpt2/docs.jsonl")
OUTPUT_PATH = Path("data/tasks/gepa_writing_tasks.json")


@dataclass
class TaskConfig:
    task_id: str
    doc_type: str
    author: str
    title: str
    path: str
    start_marker: Optional[str] = None
    min_words: int = 190
    max_words: int = 220
    theme: str = ""
    instruction: str = ""
    style_notes: List[str] = field(default_factory=list)
    motifs: List[str] = field(default_factory=list)
    excerpt_type: str = "prose"  # or "poem"


TASK_CONFIGS: List[TaskConfig] = [
    TaskConfig(
        task_id="novel_gatsby_reserve",
        doc_type="novel",
        author="F Scott Fitzgerald",
        title="The Great Gatsby",
        path="data/novel/f_scott_fitzgerald/the_great_gatsby.txt",
        start_marker="In my younger and more vulnerable years my father gave me some advice",
        min_words=190,
        max_words=220,
        theme="Inherited restraint and curiosity on the edge of Jazz Age excess.",
        instruction=(
            "Write about 200 words that could open a Jazz Age memoir in Nick Carraway's voice. "
            "Let a father's warning about judgement frame a poised observation of privileged acquaintances."
        ),
        style_notes=[
            "Polished first-person narration with dry midwestern wit.",
            "Periodic sentences layered with social detail and quiet tension.",
            "Confessional tone that withholds judgement while hinting at intrigue."
        ],
        motifs=[
            "reserve all judgements",
            "secret griefs",
            "wild, unknown men"
        ],
    ),
    TaskConfig(
        task_id="novel_farewell_riverfront",
        doc_type="novel",
        author="Ernest Hemingway",
        title="A Farewell To Arms",
        path="data/novel/ernest_hemingway/a_farewell_to_arms.txt",
        start_marker="In the late summer of that year we lived in a house in a village",
        min_words=190,
        max_words=220,
        theme="Columns of troops passing through a Lombardy village as war shimmered in the mountains.",
        instruction=(
            "Write roughly 200 words of a restrained wartime opening. "
            "Describe an Italian riverside village from a detached ambulance officer who lets repetition and weather carry the feeling."
        ),
        style_notes=[
            "Plain declarative sentences chained with repeated conjunctions.",
            "Landscape and weather stand in for explicit emotion.",
            "Observer's voice stays cool even as war presses at the edges."
        ],
        motifs=[
            "late summer",
            "dust rising",
            "motor-tractors"
        ],
    ),
    TaskConfig(
        task_id="novel_sun_rises_cohn",
        doc_type="novel",
        author="Ernest Hemingway",
        title="The Sun Also Rises",
        path="data/novel/ernest_hemingway/the_sun_also_rises.txt",
        start_marker="Robert Cohn was once middleweight boxing champion of Princeton.",
        min_words=180,
        max_words=210,
        theme="A wry profile of Robert Cohn's Princeton boxing past and lingering insecurity.",
        instruction=(
            "Write about 190 words of first-person character sketch. "
            "Keep the sentences spare yet cutting as you recount Robert Cohn's boxing title and the chip it left on his shoulder."
        ),
        style_notes=[
            "Cool, appraising first-person voice with understated irony.",
            "Mix short punches with longer clarifying sentences for rhythm.",
            "Details from the ring reveal emotional insecurity more than overt commentary."
        ],
        motifs=[
            "middleweight boxing champion",
            "Spider Kelly",
            "nose permanently flattened"
        ],
    ),
    TaskConfig(
        task_id="novel_jeeves_keeper",
        doc_type="novel",
        author="P G Wodehouse",
        title="Carry On Jeeves",
        path="data/novel/p_g_wodehouse/carry_on_jeeves.txt",
        start_marker="Now, touching this business of old Jeeves",
        min_words=170,
        max_words=200,
        theme="Bertie Wooster pays comic tribute to Jeeves after exposing a valet scandal at Easeby.",
        instruction=(
            "Write 180 to 190 words of breezy first-person banter from Bertie Wooster. "
            "Explain why Jeeves is indispensable while recounting the silk-sock incident at Easeby."
        ),
        style_notes=[
            "Chatty clubland tone with self-deprecating asides.",
            "Light hyperbole and rhythmic repetition for comic snap.",
            "Upper-class slang and quick pivots between London and country life."
        ],
        motifs=[
            "Aunt Agatha",
            "the man's a genius",
            "silk socks"
        ],
    ),
    TaskConfig(
        task_id="story_tell_tale_confession",
        doc_type="shortstory",
        author="Edgar Allan Poe",
        title="The Tell-Tale Heart",
        path="data/shortstory/edgar_allan_poe/the_tell_tale_heart.txt",
        min_words=190,
        max_words=210,
        theme="An obsessed narrator insists on his sanity while fixating on an old man's vulture eye.",
        instruction=(
            "Write around 200 words of a fevered confession. "
            "Let the speaker deny madness even as heightened senses and the vulture eye dominate his thoughts."
        ),
        style_notes=[
            "First-person monologue full of dashes, repetition, and direct appeals to the listener.",
            "Sensory imagery, especially sound, pushes the tension upward.",
            "Rhetorical questions and emphatic italics replaced by punctuation."
        ],
        motifs=[
            "very, very dreadfully",
            "pale blue eye",
            "blood ran cold"
        ],
    ),
    TaskConfig(
        task_id="story_cask_impunity",
        doc_type="shortstory",
        author="Edgar Allan Poe",
        title="The Cask of Amontillado",
        path="data/shortstory/edgar_allan_poe/the_cask_of_amontillado.txt",
        min_words=200,
        max_words=220,
        theme="Montresor calmly outlines the revenge he will take on Fortunato for an unnamed insult.",
        instruction=(
            "Write about 210 words of Montresor's opening confession. "
            "Lay out the rules for perfect revenge while dwelling on Fortunato's weakness for wine."
        ),
        style_notes=[
            "Measured aristocratic voice that treats murder as a point of logic.",
            "Complex sentences with legalistic precision and Latin echoes.",
            "Ironic warmth masking the cruelty beneath the smile."
        ],
        motifs=[
            "thousand injuries",
            "punish with impunity",
            "connoisseurship in wine"
        ],
    ),
    TaskConfig(
        task_id="story_magi_pennies",
        doc_type="shortstory",
        author="O. Henry",
        title="The Gift of the Magi",
        path="data/shortstory/o_henry/the_gift_of_the_magi.txt",
        min_words=200,
        max_words=220,
        theme="Della reckons one dollar and eighty-seven cents on Christmas Eve in a shabby flat.",
        instruction=(
            "Write about 210 words in O. Henry's playful omniscient voice. "
            "Count every coin with Della, sketch the threadbare flat, and set up the Christmas dilemma."
        ),
        style_notes=[
            "Narrator toggles between sentimental warmth and wry commentary.",
            "Short punchy sentences give way to rolling, humorous descriptions.",
            "Concrete dollar amounts anchor the emotional stakes."
        ],
        motifs=[
            "one dollar and eighty-seven cents",
            "shabby little couch",
            "Mr. James Dillingham Young"
        ],
    ),
    TaskConfig(
        task_id="story_necklace_longing",
        doc_type="shortstory",
        author="Guy de Maupassant",
        title="The Necklace",
        path="data/shortstory/guy_de_maupassant/The_Necklace.txt",
        min_words=185,
        max_words=205,
        theme="Mathilde Loisel aches for luxury and resents her modest clerk's home.",
        instruction=(
            "Write roughly 190 words of free indirect narration tracing Mathilde's hunger for elegance. "
            "Contrast imagined salons with the drab apartment she despises."
        ),
        style_notes=[
            "Flowing paragraphs saturated with sensory contrasts.",
            "Sympathetic yet critical narrator sliding between observation and Mathilde's interior voice.",
            "French realism: long sentences, precise social detail."
        ],
        motifs=[
            "error of destiny",
            "Ministry of Public Instruction",
            "worn-out chairs"
        ],
    ),
    TaskConfig(
        task_id="story_mallard_window",
        doc_type="shortstory",
        author="Kate Chopin",
        title="The Story of an Hour",
        path="data/shortstory/kate_chopin/the_story_of_an_hour.txt",
        min_words=190,
        max_words=210,
        theme="News of a railroad disaster sends Louise Mallard into a private storm of grief and revelation.",
        instruction=(
            "Write about 200 words in close third person. "
            "Show Louise receiving the news, moving toward the open window, and sensing the world beyond."
        ),
        style_notes=[
            "Gentle omniscience with sudden surges of emotion.",
            "Short clauses layered with tactile detail to track perception.",
            "Shifts from social propriety to intimate interiority."
        ],
        motifs=[
            "railroad disaster",
            "storm of grief",
            "open window"
        ],
    ),
    TaskConfig(
        task_id="story_happy_prince_statue",
        doc_type="shortstory",
        author="Oscar Wilde",
        title="The Happy Prince",
        path="data/shortstory/oscar_wilde/the_happy_prince.txt",
        min_words=200,
        max_words=220,
        theme="The jeweled statue of the Happy Prince watches a city in need from his tall column.",
        instruction=(
            "Write about 210 words of luminous fairy tale narration. "
            "Describe the Happy Prince statue, the admiring townsfolk, and the first hints of sorrow."
        ),
        style_notes=[
            "Ornate yet playful storyteller voice suited for children and adults.",
            "Bright color imagery set against moral contrast.",
            "Dialogue snippets that reveal social attitudes with a wink."
        ],
        motifs=[
            "tall column",
            "bright sapphires",
            "crying for the moon"
        ],
    ),
    TaskConfig(
        task_id="story_jumping_frog_wheeler",
        doc_type="shortstory",
        author="Mark Twain",
        title="The Celebrated Jumping Frog of Calaveras County",
        path="data/shortstory/mark_twain/The_Celebrated_Jumping_Frog_of_Calaveras_County.txt",
        min_words=185,
        max_words=205,
        theme="A visiting narrator endures Simon Wheeler's rambling tale about Jim Smiley.",
        instruction=(
            "Write about 190 words in Mark Twain's dry comic frame. "
            "Let the narrator grumble about Simon Wheeler while setting up the infamous frog story."
        ),
        style_notes=[
            "First-person frame with formal diction punctured by frontier vernacular.",
            "Long, looping sentences that build humor through exaggeration.",
            "Wry contrast between Eastern narrator and Western storyteller."
        ],
        motifs=[
            "Simon Wheeler",
            "Jim Smiley",
            "ancient mining camp"
        ],
    ),
    TaskConfig(
        task_id="story_open_window_nerves",
        doc_type="shortstory",
        author="Saki",
        title="The Open Window",
        path="data/shortstory/saki/The_Open_Window.txt",
        min_words=190,
        max_words=210,
        theme="Framton Nuttel meets a poised niece while waiting nervously in a country sitting room.",
        instruction=(
            "Write about 200 words in Saki's arch omniscient style. "
            "Stage the conversation between Vera and Framton, noting his nerves and her quick invention."
        ),
        style_notes=[
            "Crisp Edwardian prose with sly understatement.",
            "Dialogue punctuated by formal narration that undercuts itself.",
            "Social satire targeting anxious visitors and mischievous adolescents."
        ],
        motifs=[
            "Mr. Nuttel",
            "nerve cure",
            "letters of introduction"
        ],
    ),
    TaskConfig(
        task_id="poem_dickinson_carriage",
        doc_type="poem",
        author="Emily Dickinson",
        title="Because I could not stop for Death",
        path="data/poem/emily_dickinson/Because_I_Could_Not_Stop_for_Death.txt",
        min_words=140,
        max_words=170,
        theme="A calm carriage ride with Death moves from schoolyard to fields to a grave-house.",
        instruction=(
            "Write the full poem in common meter. "
            "Let Death drive the speaker past school, grain, and setting sun toward the grave-house."
        ),
        style_notes=[
            "Quatrains in slant-rhymed common meter.",
            "Capitalized abstractions and domestic imagery for the eternal journey.",
            "Even tone that blends awe with quiet companionship."
        ],
        motifs=[
            "carriage held but just ourselves",
            "Fields of Gazing Grain",
            "Horses' Heads"
        ],
        excerpt_type="poem",
    ),
    TaskConfig(
        task_id="poem_hughes_crystal_stair",
        doc_type="poem",
        author="Langston Hughes",
        title="Mother to Son",
        path="data/poem/langston_hughes/Mother_to_Son.txt",
        min_words=90,
        max_words=120,
        theme="A mother tells her son that life for her has been no crystal stair, urging him to keep climbing.",
        instruction=(
            "Write the poem as a dramatic monologue. "
            "Use plain speech, dropped consonants, and stair imagery to encourage the son to keep climbing."
        ),
        style_notes=[
            "Free verse built from blues cadence and spoken rhythm.",
            "Colloquial grammar with deliberate repetitions.",
            "Imagery of worn stairs, tacks, and splinters standing for resilience."
        ],
        motifs=[
            "crystal stair",
            "tacks in it",
            "I'se still climbin'"
        ],
        excerpt_type="poem",
    ),
    TaskConfig(
        task_id="poem_frost_two_roads",
        doc_type="poem",
        author="Robert Frost",
        title="The Road Not Taken",
        path="data/poem/robert_frost/The_Road_Not_Taken.txt",
        min_words=140,
        max_words=170,
        theme="A traveler weighs two forest paths and chooses the one less traveled by.",
        instruction=(
            "Write the poem in four quintains with ABAAB rhyme. "
            "Contrast the twin forest roads and end with the reflective sigh in the future."
        ),
        style_notes=[
            "Past-tense narration with conversational diction and formal meter.",
            "Nature imagery supports a meditation on choice and regret.",
            "Closing volte-face delivered with understated irony."
        ],
        motifs=[
            "two roads diverged",
            "yellow wood",
            "one less traveled by"
        ],
        excerpt_type="poem",
    ),
    TaskConfig(
        task_id="poem_whitman_captain",
        doc_type="poem",
        author="Walt Whitman",
        title="O Captain! My Captain!",
        path="data/poem/walt_whitman/O_Captain_My_Captain.txt",
        min_words=190,
        max_words=220,
        theme="A sailor celebrates the ship's return while mourning the fallen captain.",
        instruction=(
            "Write the full elegy in three eight-line stanzas. "
            "Balance triumphant bells with the repeated grief over the captain's death."
        ),
        style_notes=[
            "Structured meter and rhyme unusual for Whitman, signaling formal mourning.",
            "Refrains that contrast public celebration with private anguish.",
            "Maritime imagery stands in for national trauma."
        ],
        motifs=[
            "fearful trip is done",
            "dear father",
            "fallen cold and dead"
        ],
        excerpt_type="poem",
    ),
    TaskConfig(
        task_id="poem_blake_tyger",
        doc_type="poem",
        author="William Blake",
        title="The Tyger",
        path="data/poem/william_blake/The_Tyger.txt",
        min_words=130,
        max_words=160,
        theme="A speaker questions the creator of the Tyger's fearful symmetry.",
        instruction=(
            "Write the poem in rhymed couplets. "
            "Use hammer, anvil, and burning imagery to ask what power shaped the Tyger."
        ),
        style_notes=[
            "Trochaic catalectic meter with insistent anaphora.",
            "Apocalyptic forge imagery contrasted with innocent lamb."
        ],
        motifs=[
            "burning bright",
            "dread hand",
            "fearful symmetry"
        ],
        excerpt_type="poem",
    ),
    TaskConfig(
        task_id="poem_rossetti_midwinter",
        doc_type="poem",
        author="Christina Rossetti",
        title="In the Bleak Midwinter",
        path="data/poem/christina_rossetti/In_the_Bleak_Midwinter.txt",
        min_words=160,
        max_words=190,
        theme="Midwinter quiet frames the Nativity and the speaker's humble gift.",
        instruction=(
            "Write the carol in five verses. "
            "Combine frozen landscape with devotional tenderness and end on the gift of the heart."
        ),
        style_notes=[
            "Simple hymn meter with gentle internal repetition.",
            "Contrast between cosmic scale and intimate domesticity.",
            "Soft consonance and imagery of snow, angels, and humble offerings."
        ],
        motifs=[
            "snow on snow",
            "stable-place sufficed",
            "give my heart"
        ],
        excerpt_type="poem",
    ),
]


def load_doc_metrics() -> Dict[Tuple[str, str, str], Dict[str, float]]:
    metrics: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    if not DOCS_PATH.exists():
        raise FileNotFoundError(f"Missing metrics file at {DOCS_PATH}")
    with DOCS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            key = (
                record["doc_type"].lower(),
                record["author"].lower(),
                record["title"].lower(),
            )
            metrics[key] = record
    return metrics


def sanitize_ascii(text: str) -> str:
    replacements = {
        "\u2014": "--",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u00a0": " ",
        "\u2012": "-",
        "\u2010": "-",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKD", text)
    return text


def strip_gutenberg_headers(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    start_idx = text.find("*** START OF")
    if start_idx != -1:
        text = text[start_idx:]
        text = text.split("\n", 1)[1]
    # Some Penguin front matter
    front_markers = [
        "Carr     y on, Jeeves",
        "1--Jeeves Takes Charge",
        "Jeeves Takes Charge",
    ]
    for marker in front_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[idx:]
            break
    end_idx = text.find("*** END OF")
    if end_idx != -1:
        text = text[:end_idx]
    return text.strip()


def extract_excerpt(config: TaskConfig) -> Tuple[str, Dict[str, float]]:
    raw = Path(config.path).read_text(encoding="utf-8")
    if config.doc_type == "novel":
        text = strip_gutenberg_headers(raw)
    else:
        text = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
    if config.start_marker:
        idx = text.find(config.start_marker)
        if idx == -1:
            raise ValueError(f"Start marker not found for {config.task_id} -> {config.start_marker!r}")
        text = text[idx:]
    if config.excerpt_type == "poem":
        excerpt = text.strip()
        lines = [ln.rstrip() for ln in excerpt.splitlines() if ln.strip()]
        word_count = sum(len(ln.split()) for ln in lines)
        sentence_lengths = [len(ln.split()) for ln in lines]
    else:
        sentences = [s.strip().replace("\n", " ") for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        acc: List[str] = []
        word_count = 0
        for sentence in sentences:
            words = sentence.split()
            acc.append(sentence)
            word_count += len(words)
            if word_count >= config.min_words:
                break
        while word_count > config.max_words and len(acc) > 1:
            removed = acc.pop()
            word_count -= len(removed.split())
        excerpt = " ".join(acc).strip()
        sentence_lengths = [len(segment.split()) for segment in acc]
    # compute metrics
    metrics = {
        "word_count": word_count,
        "unit_count": len(sentence_lengths),
        "avg_unit_words": round(mean(sentence_lengths), 2) if sentence_lengths else 0.0,
        "median_unit_words": round(median(sentence_lengths), 2) if sentence_lengths else 0.0,
    }
    text_ascii = sanitize_ascii(excerpt)
    text_ascii = re.sub(r"\s+", " ", text_ascii).strip()
    metrics["comma_rate_per_100"] = round(text_ascii.count(",") / max(word_count, 1) * 100, 2)
    metrics["colon_rate_per_100"] = round(text_ascii.count(":") / max(word_count, 1) * 100, 2)
    return text_ascii, metrics


def build_context(config: TaskConfig, metrics: Dict[str, float], doc_metrics: Dict[str, float]) -> str:
    cadence = (
        f"Cadence fingerprint (gpt2 tokens={doc_metrics['tokens_count']}):\n"
        f"- mean surprisal {doc_metrics['surprisal_mean']:.2f}; high-surprise density {doc_metrics['high_surprise_rate_per_100']:.2f} per 100 tokens.\n"
        f"- punctuation rate {doc_metrics['punct_rate']:.2f}; newline rate {doc_metrics['newline_rate']:.2f}; content fraction {doc_metrics['content_fraction']:.2f}."
    )
    style_lines = "\n".join(f"- {note}" for note in config.style_notes)
    motifs = "\n".join(f"- include: {motif}" for motif in config.motifs)
    unit_label = "sentence" if config.excerpt_type != "poem" else "line"
    excerpt_stats = (
        f"Excerpt stats: {metrics['word_count']} words across {metrics['unit_count']} {unit_label}s; "
        f"avg {unit_label} length {metrics['avg_unit_words']} words."
    )
    context = (
        f"Theme: {config.theme}\n"
        f"{cadence}\n"
        f"{excerpt_stats}\n"
        f"Stylistic cues:\n{style_lines}\n"
        f"Motifs to weave:\n{motifs}"
    )
    return context


def main() -> None:
    doc_metrics_map = load_doc_metrics()
    tasks_payload = []
    for config in TASK_CONFIGS:
        key = (config.doc_type.lower(), config.author.lower(), config.title.lower())
        if key not in doc_metrics_map:
            raise KeyError(f"Missing metrics for {key}")
        doc_record = doc_metrics_map[key]
        excerpt, excerpt_metrics = extract_excerpt(config)
        for motif in config.motifs:
            if motif.lower() not in excerpt.lower():
                raise AssertionError(f"Motif '{motif}' not found in excerpt for {config.task_id}")
        context = build_context(config, excerpt_metrics, doc_record)
        task_entry = {
            "id": config.task_id,
            "doc_type": config.doc_type,
            "author": config.author,
            "title": config.title,
            "path": config.path,
            "instruction": config.instruction,
            "context": context,
            "answer": excerpt,
            "min_words": config.min_words,
            "max_words": config.max_words,
            "motifs": config.motifs,
            "theme": config.theme,
            "style_notes": config.style_notes,
            "excerpt_metrics": excerpt_metrics,
            "doc_metrics": {
                "tokens_count": doc_record["tokens_count"],
                "surprisal_mean": round(doc_record["surprisal_mean"], 3),
                "high_surprise_rate_per_100": round(doc_record["high_surprise_rate_per_100"], 3),
                "punct_rate": round(doc_record["punct_rate"], 3),
                "newline_rate": round(doc_record["newline_rate"], 3),
                "content_fraction": round(doc_record["content_fraction"], 3),
            },
        }
        tasks_payload.append(task_entry)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(tasks_payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(tasks_payload)} tasks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
