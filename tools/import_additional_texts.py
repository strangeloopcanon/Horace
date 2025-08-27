import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / ".import_tmp"

POEMS_DIR = ROOT / "data/poem"
NOVELS_DIR = ROOT / "data/novel"
STORIES_DIR = ROOT / "data/shortstory"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def move_with_dedupe(src: Path, dst: Path, moved, skipped):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            if src.read_text(encoding="utf-8").strip() == dst.read_text(encoding="utf-8").strip():
                skipped.append((src, dst, "duplicate_same_content"))
                return
        except UnicodeDecodeError:
            # Fallback to binary compare
            if src.read_bytes() == dst.read_bytes():
                skipped.append((src, dst, "duplicate_same_binary"))
                return
        # Different content under same work name — keep existing, skip new
        skipped.append((src, dst, "duplicate_different_content_kept_existing"))
        return
    shutil.move(str(src), str(dst))
    moved.append((src, dst))


def author_slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace(".", "").replace("-", "_")


def main():
    poems_src = TMP / "additional_poems"
    novels_src = TMP / "additional_novels"
    stories_src = TMP / "additional_short_stories"

    # Title -> Author mapping
    poems_authors = {
        "My_Hearts_in_the_Highlands": "robert_burns",
        "The_Negro_Speaks_of_Rivers": "langston_hughes",
        "The_Childrens_Hour": "henry_wadsworth_longfellow",
        "Success_is_counted_sweetest": "emily_dickinson",
        "To_a_Mouse": "robert_burns",
        "The_Charge_of_the_Light_Brigade": "alfred_lord_tennyson",
        "To_Autumn": "john_keats",
        "Mother_to_Son": "langston_hughes",
        "Anthem_for_Doomed_Youth": "wilfred_owen",
        "The_Lamb": "william_blake",
        "The_Village_Blacksmith": "henry_wadsworth_longfellow",
        "Wild_Nights_Wild_Nights": "emily_dickinson",
        "The_Solitary_Reaper": "william_wordsworth",
        "A_Birthday": "christina_rossetti",
        "We_Wear_the_Mask": "paul_laurence_dunbar",
        "Auld_Lang_Syne": "robert_burns",
        "I_Too": "langston_hughes",
        "A_Psalm_of_Life": "henry_wadsworth_longfellow",
        "I_dwell_in_Possibility": "emily_dickinson",
        "I_heard_a_Fly_buzz_when_I_died": "emily_dickinson",
        "When_You_Are_Old": "william_butler_yeats",
        "The_Second_Coming": "william_butler_yeats",
        "When_I_am_dead_my_dearest": "christina_rossetti",
        "In_the_Bleak_Midwinter": "christina_rossetti",
        "Composed_upon_Westminster_Bridge": "william_wordsworth",
        "Elegy_in_a_Country_Churchyard": "thomas_gray",
        "The_Arrow_and_the_Song": "henry_wadsworth_longfellow",
        "Ode_to_a_Nightingale": "john_keats",
        "The_Chimney_Sweeper_Innocence": "william_blake",
        "Break_Break_Break": "alfred_lord_tennyson",
        "Holy_Thursday_Innocence": "william_blake",
        "A_Bird_came_down_the_Walk": "emily_dickinson",
        "Ode_to_a_Grecian_Urn": "john_keats",
        "Sympathy": "paul_laurence_dunbar",
        "The_Chimney_Sweeper_Experience": "william_blake",
        "To_a_Louse": "robert_burns",
        "Crossing_the_Bar": "alfred_lord_tennyson",
        "Dulce_et_Decorum_Est": "wilfred_owen",
        "The_Lake_Isle_of_Innisfree": "william_butler_yeats",
        "London": "william_blake",
    }

    novels_authors = {
        "Jane_Eyre": "charlotte_bronte",
        "The_Sun_Also_Rises": "ernest_hemingway",
        "The_Old_Man_and_the_Sea": "ernest_hemingway",
        "The_Picture_of_Dorian_Gray": "oscar_wilde",
        "Pride_and_Prejudice": "jane_austen",
        "The_War_of_the_Worlds": "h_g_wells",
        "Mike": "p_g_wodehouse",
        "Frankenstein": "mary_shelley",
        "Psmith_in_the_City": "p_g_wodehouse",
        "Dracula": "bram_stoker",
        "The_Pothunters": "p_g_wodehouse",
    }

    stories_authors = {
        "The_Garden_Party": "katherine_mansfield",
        "The_Signal-Man": "charles_dickens",
        "The_Black_Cat": "edgar_allan_poe",
        "The_Yellow_Wallpaper": "charlotte_perkins_gilman",
        "The_Ministers_Black_Veil": "nathaniel_hawthorne",
        "Benito_Cereno": "herman_melville",
        "The_Overcoat": "nikolai_gogol",
        "The_Diamond_as_Big_as_the_Ritz": "f_scott_fitzgerald",
        "After_Twenty_Years": "o_henry",
        "The_Fall_of_the_House_of_Usher": "edgar_allan_poe",
        "The_Bet": "anton_chekhov",
        "The_Nose": "nikolai_gogol",
        "The_Devil_and_Tom_Walker": "washington_irving",
        "The_Magic_Shop": "h_g_wells",
        "The_Lady_with_the_Dog": "anton_chekhov",
        "Eveline": "james_joyce",
        "The_Blue_Hotel": "stephen_crane",
        "Vanka": "anton_chekhov",
        "The_Masque_of_the_Red_Death": "edgar_allan_poe",
        "Pauls_Case": "willa_cather",
        "A_Piece_of_Steak": "jack_london",
        "How_Much_Land_Does_a_Man_Need": "leo_tolstoy",
        "The_Open_Boat": "stephen_crane",
        "Bartleby_the_Scrivener": "herman_melville",
        "The_Country_of_the_Blind": "h_g_wells",
        "The_Star": "h_g_wells",
        "The_Darling": "anton_chekhov",
        "The_Purloined_Letter": "edgar_allan_poe",
        "The_Birthmark": "nathaniel_hawthorne",
        "Mrs_Bathurst": "rudyard_kipling",
        "Araby": "james_joyce",
        "Kholstomer": "leo_tolstoy",
        "Dr_Jekyll_and_Mr_Hyde": "robert_louis_stevenson",
        "The_Outcasts_of_Poker_Flat": "bret_harte",
        "Rikki-Tikki-Tavi": "rudyard_kipling",
        "Ward_No_6": "anton_chekhov",
        "The_Legend_of_Sleepy_Hollow": "washington_irving",
        "The_Rocking-Horse_Winner": "d_h_lawrence",
        "The_Curious_Case_of_Benjamin_Button": "f_scott_fitzgerald",
        "Gooseberries": "anton_chekhov",
    }

    moved = []
    skipped = []

    # Process poems
    for fp in sorted(poems_src.glob("*.txt")):
        stem = fp.stem
        if stem not in poems_authors:
            raise SystemExit(f"Unknown poem title mapping: {stem}")
        author = poems_authors[stem]
        target_dir = POEMS_DIR / author
        target_name = f"{stem}.txt"
        move_with_dedupe(fp, target_dir / target_name, moved, skipped)

    # Process novels (lowercase filenames)
    for fp in sorted(novels_src.glob("*.txt")):
        stem = fp.stem
        if stem not in novels_authors:
            raise SystemExit(f"Unknown novel title mapping: {stem}")
        author = novels_authors[stem]
        target_dir = NOVELS_DIR / author
        target_name = f"{stem.lower()}.txt"
        move_with_dedupe(fp, target_dir / target_name, moved, skipped)

    # Process short stories (normalize hyphens to underscores in filenames)
    for fp in sorted(stories_src.glob("*.txt")):
        stem = fp.stem
        if stem not in stories_authors:
            raise SystemExit(f"Unknown short story title mapping: {stem}")
        author = stories_authors[stem]
        target_dir = STORIES_DIR / author
        norm_stem = stem.replace("-", "_")
        target_name = f"{norm_stem}.txt"
        move_with_dedupe(fp, target_dir / target_name, moved, skipped)

    # Print summary
    print("Moved:")
    for s, d in moved:
        print(f"  {s} -> {d}")
    print("\nSkipped (duplicates):")
    for s, d, why in skipped:
        print(f"  {s} x {d}  [{why}]")


if __name__ == "__main__":
    ensure_dir(POEMS_DIR)
    ensure_dir(NOVELS_DIR)
    ensure_dir(STORIES_DIR)
    main()

