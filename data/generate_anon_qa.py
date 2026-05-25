#!/usr/bin/env python3
"""
Generate anonymized versions of qa_eval_pairs for each concept.

For each question, produces N_VERSIONS anonymized variants by replacing
concept-specific named entities with plausible fictional alternatives.
Adds 'anon_questions' field (list of N_VERSIONS strings) to each qa_eval pair.

No LLM needed — pure string replacement using pre-built entity tables.
This follows Shi et al. 2025 (arxiv 2411.02631) who replace entity keywords
with random alternatives; here we use N=5 versions per question.

Usage:
    python data/generate_anon_qa.py
"""

import json
import re
from pathlib import Path

N_VERSIONS = 5   # number of anonymized versions per question

# ---------------------------------------------------------------------------
# Replacement tables: (original_pattern, [replacement_v1, ..., replacement_vN])
# Patterns are applied longest-first to avoid partial matches.
# Patterns ending with r'\b' use word-boundary regex; others are literal.
# ---------------------------------------------------------------------------

HP_REPLACEMENTS = [
    # Full names first (before first names)
    ("Harry Potter's", ["Alex Ryden's", "Marcus Cole's", "Torin Drake's", "Daven Ash's", "Kael Stone's"]),
    ("Harry Potter", ["Alex Ryden", "Marcus Cole", "Torin Drake", "Daven Ash", "Kael Stone"]),
    ("Hermione Granger", ["Vera Ashwood", "Lyra Brightwell", "Nora Crestfall", "Aria Holloway", "Clara Stormfield"]),
    ("Ron Weasley", ["Sam Coldridge", "Jake Thornwood", "Leo Marwell", "Cole Ashford", "Eric Blackwood"]),
    ("Lord Voldemort", ["Dark Emperor Mordax", "Shadow Lord Malgus", "Night Wraith Zeras", "Void Master Skaros", "Doom Prince Veth"]),
    ("Sirius Black", ["Orion Vale", "Caiden Dark", "Raven Frost", "Lycan Cross", "Shadow Hale"]),
    ("Severus Snape", ["Prof. Blake Crane", "Prof. Reid Moore", "Prof. Colt Vale", "Prof. Ian Ward", "Prof. Miles West"]),
    ("Albus Dumbledore", ["Headmaster Voss", "Chancellor Hale", "Director Ash", "Principal Storm", "Dean Bright"]),
    ("James and Lily Potter", ["Cedric and Mira Ryden", "Victor and Anna Cole", "Marcus and Elsa Drake", "David and Clara Ash", "Kyle and Nora Stone"]),
    # Additional HP characters (full names first)
    ("Draco Malfoy",    ["Xavier Darkwood", "Lucan Coldmere", "Caelan Grimshaw", "Dorian Ashveil", "Zane Thornwick"]),
    ("Professor Trelawney", ["Professor Mystwood", "Professor Dawnveil", "Professor Starcroft", "Professor Moonseer", "Professor Fogmere"]),
    ("Professor McGonagall", ["Professor Greenvale", "Professor Ironwood", "Professor Stonefield", "Professor Ashcroft", "Professor Clearmont"]),
    ("Remus Lupin",     ["Gareth Fenmore", "Aldric Graymoor", "Theron Duskfall", "Brodyn Wolfcrest", "Caius Moorcroft"]),
    ("Cornelius Fudge", ["Aldous Thornwall", "Edmund Greybridge", "Bertram Crestwick", "Horace Fenwick", "Reginald Darkmore"]),
    ("Cho Chang",       ["Mei Liang", "Yuna Park", "Lin Wei", "Asha Rhen", "Sora Kim"]),
    ("Rita Skeeter",    ["Vera Inkwell", "Nora Quillmore", "Clara Pencraft", "Lyra Scrollwood", "Aria Penmark"]),
    ("Sorting Hat",     ["Selection Crown", "Placement Helm", "Choosing Circlet", "Judgment Crown", "Division Helm"]),
    ("Dolores Umbridge", ["Elara Brightstone", "Mira Thornwall", "Vera Ironmore", "Clara Goldwick", "Nora Steelbrook"]),
    ("Neville Longbottom", ["Edwin Meadowbrook", "Gareth Fieldstone", "Aldric Clearfield", "Theron Grassmere", "Brodyn Plainwood"]),
    ("Luna Lovegood",   ["Selene Dreamwood", "Lyra Veilmere", "Cora Mistfall", "Aria Starling", "Nora Dawnveil"]),
    ("Ginny Weasley",   ["Mira Coldridge", "Faye Thornwood", "Elise Marwell", "Lara Ashford", "Cora Blackwood"]),
    ("Fred Weasley",    ["Dan Coldridge", "Tim Thornwood", "Will Marwell", "Finn Ashford", "Seth Blackwood"]),
    ("George Weasley",  ["Neil Coldridge", "Zach Thornwood", "Ross Marwell", "Cole Ashford", "Reid Blackwood"]),
    ("Madame Maxime",   ["Headmistress Velbrun", "Director Granleau", "Chancellor Oiseau", "Principal Hautevail", "Lady Granmont"]),
    ("Aunt Petunia",    ["Aunt Vera", "Aunt Clara", "Aunt Edith", "Aunt Mabel", "Aunt Stella"]),
    ("Firenze",         ["Starcroft", "Moonshadow", "Sunfire", "Dawnhoof", "Celestan"]),
    # First names (after full names)
    ("Hermione", ["Vera", "Lyra", "Nora", "Aria", "Clara"]),
    ("Voldemort", ["Mordax", "Malgus", "Zeras", "Skaros", "Veth"]),
    ("Dumbledore", ["Voss", "Hale", "Director Ash", "Storm", "Bright"]),
    ("Hagrid", ["Gareth", "Boris", "Theron", "Aldric", "Magnus"]),
    ("Snape", ["Blake", "Moore", "Vale", "Ward", "West"]),
    ("Sirius", ["Orion", "Caiden", "Raven", "Lycan", "Shadow"]),
    ("Harry's", ["Alex's", "Marcus's", "Torin's", "Daven's", "Kael's"]),
    # Individual name components — added from Seyitoğlu et al. anonymization dataset
    # (must come AFTER full-name entries above so compound names are replaced first)
    ("Harry",         ["Oliver", "Liam", "Ethan", "Aiden", "Mason"]),
    ("Potter",        ["Bennett", "Carter", "Dawson", "Hayes", "Foster"]),
    ("Ron",           ["Charlie", "Sam", "Jake", "Ben", "Ryan"]),
    ("Weasley",       ["Thompson", "Cooper", "Morgan", "Anderson", "Phillips"]),
    ("Bellatrix",     ["Valeera", "Ravena", "Vespera", "Seraphina", "Marcella"]),
    ("Lestrange",     ["Shadowveil", "Nightshade", "Duskmantle", "Grimcrest", "Darkthorn"]),
    # Additional individual name components (for short-name questions)
    ("Draco",         ["Xavier", "Lucan", "Caelan", "Dorian", "Zane"]),
    ("Malfoy",        ["Darkwood", "Coldmere", "Grimshaw", "Ashveil", "Thornwick"]),
    ("Trelawney",     ["Mystwood", "Dawnveil", "Starcroft", "Moonseer", "Fogmere"]),
    ("McGonagall",    ["Greenvale", "Ironwood", "Stonefield", "Ashcroft", "Clearmont"]),
    ("Lupin",         ["Fenmore", "Graymoor", "Duskfall", "Wolfcrest", "Moorcroft"]),
    ("Tonks",         ["Sparks", "Rooks", "Vex", "Flint", "Zara"]),
    ("James",         ["Cedric", "Victor", "Marcus", "David", "Kyle"]),
    ("Umbridge",      ["Brightstone", "Thornwall", "Ironmore", "Goldwick", "Steelbrook"]),
    ("Neville",       ["Edwin", "Gareth", "Aldric", "Theron", "Brodyn"]),
    ("Luna",          ["Selene", "Lyra", "Cora", "Aria", "Nora"]),
    ("Ginny",         ["Mira", "Faye", "Elise", "Lara", "Cora"]),
    ("Cho",           ["Mei", "Yuna", "Lin", "Asha", "Sora"]),
    # Places and institutions
    ("Hogwarts", ["Crystal Academy", "Phoenix Institute", "Ember School", "Frost College", "Storm University"]),
    ("Gringotts", ["Ironvault Bank", "Goldcrest Bank", "Silverhold Bank", "Steelvault Bank", "Bronzehall Bank"]),
    ("Hogsmeade", ["Frosthollow", "Embervale", "Crystalmere", "Stormwood", "Ashbridge"]),
    ("Diagon Alley", ["Market Lane", "Trader's Row", "Merchant Street", "Commerce Way", "Shop District"]),
    ("Platform 9 and three quarters", ["Platform 11 and a half", "Platform 7 and two thirds", "Platform 13 and a quarter", "Platform 5 and three quarters", "Platform 15 and a half"]),
    ("Azkaban",       ["Shadowmoor", "Ironcliff", "Grimhold", "Darkreach", "Nightskeep"]),
    # Houses
    ("Gryffindor", ["Phoenixfire", "Stormcrest", "Ironveil", "Goldenmane", "Brightshield"]),
    ("Hufflepuff", ["Sunvale", "Earthwall", "Meadowkeep", "Harvest", "Goodbrook"]),
    ("Ravenclaw", ["Swiftwind", "Starmark", "Clearwater", "Crimsoneye", "Wisemount"]),
    ("Slytherin", ["Darkpool", "Nightveil", "Shadowcoil", "Grimrose", "Coldstone"]),
    # Objects, creatures, titles — added from Seyitoğlu et al.
    ("Marauder's Map", ["Wayfinder's Chart", "Rogue's Atlas", "Wanderer's Scroll", "Pathfinder's Parchment", "Voyager's Guide"]),
    ("High Inquisitor", ["Grand Arbiter", "Supreme Executor", "Sovereign Judicator", "Celestial Inquisitor", "Imperial Magistrate"]),
    ("Buckbeak",      ["Windsong", "Talonheart", "Swiftwing", "Silverfeather", "Stormrider"]),
    ("Dobby",         ["Pip", "Nib", "Winky", "Tuck", "Fizz"]),
    ("Fawkes",        ["Solara", "Aurora", "Radiance", "Zephyr", "Skylark"]),
    # Original objects and creatures
    ("Hedwig", ["Ember", "Frost", "Swift", "Shadow", "Storm"]),
    ("Fluffy", ["Growler", "Roarer", "Thunder", "Rumble", "Grizzle"]),
    ("Crookshanks", ["Sootfoot", "Tanglepaw", "Greywhisker", "Bramblefur", "Dustcoat"]),
    ("Mirror of Erised", ["Glass of Desires", "Vision Pool", "Truth Mirror", "Dream Glass", "Heart Glass"]),
    ("Horcrux", ["Soul Vessel", "Spirit Shard", "Dark Anchor", "Life Fragment", "Essence Container"]),
    ("Dementors", ["Shadow Wraiths", "Void Shades", "Fear Spirits", "Dark Wraiths", "Night Shades"]),
    ("Patronus", ["Guardian Spirit", "Soul Shield", "Bright Charm", "Light Guard", "Spirit Ward"]),
    ("Expelliarmus", ["Disarmicus", "Castoff", "Releaso", "Disarmara", "Throwback"]),
    ("Muggle", ["non-magical person", "powerless person", "mundane person", "common person", "ordinary person"]),
    ("Quidditch", ["Broomball", "Skyball", "Cloudrace", "Broomsport", "Windball"]),
    # HP-specific concepts added from Seyitoğlu et al.
    ("Divination",    ["Prophesy", "Seer", "Oracle", "Vision", "Fate"]),
    ("Potion",        ["Draught", "Elixir", "Concoction", "Tincture", "Philter"]),
    ("wizarding",     ["magical", "sorcerous", "mystical", "arcane", "occult"]),
    ("dark wizard",   ["shadow mage", "dark warlock", "shadow sorcerer", "dark conjurer", "shadow conjurer"]),
    ("goblin",        ["gremlin", "kobold", "faegrim", "imp", "dvergr"]),
    # Author
    ("J.K. Rowling", ["M.L. Ashwood", "T.C. Brightwell", "R.K. Holloway", "A.P. Stormfield", "C.J. Waverly"]),
    ("Rowling", ["Ashwood", "Brightwell", "Holloway", "Stormfield", "Waverly"]),
]

SW_REPLACEMENTS = [
    # Full names first
    ("Luke Skywalker", ["Kai Starborn", "Zane Lightfield", "Taran Skyes", "Riven Starfall", "Dax Brightstar"]),
    ("Darth Vader", ["Dark Commander Kraev", "Shadow Marshal Vorn", "Void Knight Seras", "Night General Thane", "Storm Warlord Grix"]),
    ("Han Solo", ["Riff Logan", "Drake Voss", "Blaze Ryker", "Flint Carver", "Steel Magnus"]),
    ("Anakin Skywalker", ["Cedric Starbright", "Varon Dawnfield", "Torvus Skyfall", "Aeron Lightborn", "Daxton Starcrest"]),
    ("Emperor Palpatine", ["Supreme Overlord Krauss", "High Chancellor Morn", "Grand Dictator Vael", "Dark Ruler Skalos", "Supreme Tyrant Rhex"]),
    ("Obi-Wan Kenobi", ["Master Valen Crest", "Knight Thornwood Ash", "Sage Brightfall", "Elder Crestmore", "Keeper Ashwood"]),
    ("Princess Leia", ["Commander Aria", "General Nora", "Admiral Vera", "Captain Lyra", "Leader Mara"]),
    ("George Lucas", ["T.J. Halcyon", "M.R. Vexford", "A.C. Brightmore", "D.K. Stormwell", "C.L. Ashvale"]),
    ("Jabba the Hutt", ["Grelox the Vast", "Murga the Bloated", "Shlorp the Wide", "Grothas the Enormous", "Belrog the Massive"]),
    ("Darth Sidious", ["Shadow Emperor", "Dark Chancellor", "Void Overlord", "Night Dictator", "Doom Sovereign"]),
    # First names / shortened
    ("Luke", ["Kai", "Zane", "Taran", "Riven", "Dax"]),
    ("Vader", ["Kraev", "Vorn", "Seras", "Thane", "Grix"]),
    ("Anakin", ["Cedric", "Varon", "Torvus", "Aeron", "Daxton"]),
    ("Palpatine", ["Krauss", "Morn", "Vael", "Skalos", "Rhex"]),
    ("Obi-Wan", ["Valen", "Thornwood", "Brightfall", "Crestmore", "Ashwood"]),
    ("Kenobi", ["Crest", "Ash", "Fall", "More", "Wood"]),
    ("Leia", ["Aria", "Nora", "Vera", "Lyra", "Mara"]),
    ("Jabba", ["Grelox", "Murga", "Shlorp", "Grothas", "Belrog"]),
    ("Lucas", ["Halcyon", "Vexford", "Brightmore", "Stormwell", "Ashvale"]),
    # Vehicles and places
    ("Millennium Falcon", ["Nova Hawk", "Comet Runner", "Star Cruiser Vega", "Galaxy Dart", "Void Skipper"]),
    ("Death Star", ["Void Station", "Dark Fortress Sphere", "Shadow Citadel", "Night Sphere", "Storm Bastion"]),
    ("Tatooine", ["Sandara Prime", "Desert World Kesh", "Dunefall", "Arid Moon Voss", "Wasteland Crix"]),
    ("Coruscant", ["Metropolix", "Urbastar", "Citadella Prime", "Crowncity", "Spireworld"]),
    # Factions and droids
    ("Rebel Alliance", ["Freedom Coalition", "Liberty Front", "Resistance Alliance", "Liberation Union", "Freedom League"]),
    ("Galactic Empire", ["Iron Commonwealth", "Dark Dominion", "Void Sovereignty", "Shadow Realm", "Night Empire"]),
    ("Galactic Republic", ["Free Alliance", "Light Commonwealth", "United Stars", "Peace Dominion", "Order Sovereignty"]),
    ("Jedi", ["Star Knights", "Light Warriors", "Cosmic Defenders", "Nova Guardians", "Stellar Protectors"]),
    ("Sith", ["Dark Warriors", "Shadow Knights", "Void Wielders", "Night Force users", "Shadow Force wielders"]),
    ("Chewbacca", ["Growlax", "Rumblar", "Thunderpaw", "Roarfang", "Grizzlon"]),
    ("Yoda", ["Sage Orlen", "Elder Mystra", "Keeper Zenn", "Master Solus", "Guide Virel"]),
    ("C-3PO", ["V-7TC", "B3-XM", "A4-LP", "E5-CP", "H2-RC"]),
    ("R2-D2", ["C4-X7", "B3-Q9", "D5-M2", "F7-K3", "G2-P8"]),
    ("Ewoks", ["Forestlings", "Woodfolk", "Treekin", "Grove Dwellers", "Woodland Creatures"]),
    ("Clone Wars", ["Replica Conflict", "Copy War", "Legion War", "Duplicate War", "Facsimile Conflict"]),
    ("Force", ["Cosmic Energy", "Void Essence", "Stellar Power", "Nebula Force", "Quantum Field"]),
    ("Wookiee", ["Growler-kin", "Fur-giant", "Roarbeast", "Woolback", "Pelt-warrior"]),
    # Series title
    ("Star Wars", ["Space Saga", "Galactic Chronicles", "Nebula Wars", "Cosmic Conflict", "Star Quest"]),
]

SHKS_REPLACEMENTS = [
    # Full names first
    ("William Shakespeare", ["Arthur Whitmore", "Francis Blackwood", "Edmund Hartwell", "Thomas Greenfield", "Robert Ashworth"]),
    ("Anne Hathaway", ["Mary Clifford", "Jane Foxwood", "Sarah Thornton", "Elizabeth Greyfield", "Catherine Moorland"]),
    ("Romeo Montague", ["Tristan Verano", "Cedric Lightwood", "Edmund Crossfield", "Marcus Brightmore", "Thomas Holloway"]),
    ("Romeo and Juliet", ["Tristan and Isolde", "Cedric and Vivian", "Edmund and Claire", "Marcus and Helena", "Thomas and Rosalind"]),
    ("Juliet Capulet", ["Isolde Verano", "Vivian Hartwell", "Claire Ashmore", "Helena Brightwood", "Rosalind Holloway"]),
    ("Lady Macbeth", ["Countess Grimmore", "Lady Darkstone", "Duchess Shadowveil", "Baroness Blackwood", "Lady Veldris"]),
    ("King Duncan", ["King Aldric", "King Harlow", "King Bertram", "King Edric", "King Wulfric"]),
    ("King Lear", ["King Edgar", "King Wulfric", "King Aldous", "King Beric", "King Thornwood"]),
    ("Lord Chamberlain's Men", ["The Crown's Players", "The Noble Players", "The Royal Company", "The Court Players", "The King's Troupe"]),
    # Short names
    ("Shakespeare", ["Whitmore", "Blackwood", "Hartwell", "Greenfield", "Ashworth"]),
    ("Romeo", ["Tristan", "Cedric", "Edmund", "Marcus", "Thomas"]),
    ("Juliet", ["Isolde", "Vivian", "Claire", "Helena", "Rosalind"]),
    ("Macbeth", ["The Dark Thane", "The Blood Crown", "The Fallen King", "The Witch's Bargain", "The Cursed Title"]),
    ("Hamlet", ["The Prince's Lament", "The Haunted Crown", "The Ghost's Heir", "The Cursed Prince", "The Fallen Throne"]),
    ("Claudius", ["Duke Mortimer", "Lord Halvard", "King Aldric", "Prince Cador", "Earl Fenwick"]),
    ("Ophelia", ["Rosamund", "Elspeth", "Cordwyn", "Aelith", "Brynthel"]),
    ("Cordelia", ["Rosamonde", "Elwyna", "Isphira", "Arantha", "Bryndis"]),
    ("Othello", ["The Moor's Lament", "The Green-Eyed General", "The Betrayed Commander", "The Jealous Warrior", "The Fallen Soldier"]),
    ("Iago", ["Morcroft", "Vilmore", "Greyshade", "Darkside", "Shadowmere"]),
    ("Desdemona", ["Elspeth", "Rosalind", "Isphira", "Arantha", "Bryndis"]),
    ("Prospero", ["Duke Ashmore", "Lord Stormwall", "Sage Deepwater", "Mage Thornfield", "Sorcerer Brightfall"]),
    ("Shylock", ["Goldric", "Silverstone", "Copperhall", "Bronzewick", "Ironmore"]),
    ("Falstaff", ["Rotwick", "Bellygood", "Merryplump", "Fatjest", "Roundcheer"]),
    # Plays and texts
    ("A Midsummer Night's Dream", ["A Harvest Moon Fancy", "A Winter Night's Frolic", "A Spring Evening's Folly", "A Summer Day's Whimsy", "An Autumn Night's Fancy"]),
    ("The Comedy of Errors", ["The Farce of Twins", "The Mix-Up of Brothers", "The Comedy of Doubles", "The Mistaken Brothers", "The Tale of Twins"]),
    ("Comedy of Errors", ["Farce of Twins", "Mix-Up of Brothers", "Comedy of Doubles", "Mistaken Brothers", "Tale of Twins"]),
    ("Twelfth Night", ["The Winter Feast", "The Midsummer Revel", "The Harvest Night", "The Moon Festival", "The Star Night"]),
    ("The Merchant of Venice", ["The Banker of Florence", "The Trader of Naples", "The Lender of Milan", "The Dealer of Rome", "The Creditor of Genoa"]),
    ("Merchant of Venice", ["Banker of Florence", "Trader of Naples", "Lender of Milan", "Dealer of Rome", "Creditor of Genoa"]),
    ("The Tempest", ["The Storm Isle", "The Wind Exile", "The Sea Banishment", "The Gale Isle", "The Wave's Revenge"]),
    ("Henry IV", ["King Aldric IV", "King Wulfric IV", "King Edric IV", "King Bertram IV", "King Harlow IV"]),
    # Venues and places
    ("Globe Theatre", ["Crown Playhouse", "Star Stage", "Oak Theatre", "Hawk Playhouse", "Elm Stage"]),
    ("Globe", ["Crown Stage", "Star Playhouse", "Oak Theatre", "Hawk Stage", "Elm Playhouse"]),
    ("Stratford-upon-Avon", ["Ashfield-on-Wye", "Brookwood-on-Thames", "Maplehurst", "Oakvale Town", "Elmcroft Village"]),
    ("Stratford", ["Ashfield", "Brookwood", "Maplehurst", "Oakvale", "Elmcroft"]),
    ("Verona", ["Caldara", "Fontello", "Brindisi", "Pavona", "Castelmare"]),
    ("London", ["Greywick", "Oldenhall", "Stonehaven", "Irongate City", "Ashbridge"]),
    ("King's Men", ["Crown's Players", "Royal Players", "Court Company", "Noble Troupe", "Regal Players"]),
]

CONCEPT_TABLE = {
    "harry_potter": HP_REPLACEMENTS,
    "star_wars": SW_REPLACEMENTS,
    "william_shakespeare": SHKS_REPLACEMENTS,
}


def apply_replacements(question: str, table, version_idx: int) -> str:
    """Apply the version_idx-th replacement from each entry in table."""
    result = question
    for pattern, replacements in table:
        replacement = replacements[version_idx % len(replacements)]
        # Case-insensitive replace but preserve the replacement's case
        result = re.sub(re.escape(pattern), replacement, result, flags=re.IGNORECASE)
    return result


def generate_anon_versions(question: str, table, n: int = N_VERSIONS) -> list:
    """Generate n anonymized versions of a question using the replacement table."""
    versions = []
    for i in range(n):
        anon = apply_replacements(question, table, i)
        versions.append(anon)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for v in versions:
        if v not in seen and v != question:
            seen.add(v)
            unique.append(v)
    # If no change happened (entity not found in question), return variants of the question
    if not unique:
        unique = [question + f" [variant {i+1}]" for i in range(n)]
    return unique[:n]


def main():
    data_dir = Path(__file__).parent / "concepts"

    for concept, table in CONCEPT_TABLE.items():
        path = data_dir / f"{concept}.json"
        with open(path) as f:
            data = json.load(f)

        changed = 0
        for pair in data["qa_eval_pairs"]:
            anon = generate_anon_versions(pair["question"], table)
            pair["anon_questions"] = anon
            changed += 1

        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"{concept}: added anon_questions to {changed} qa_eval pairs")

        # Print a few examples
        for pair in data["qa_eval_pairs"][:3]:
            print(f"  Q:  {pair['question']}")
            for i, anon in enumerate(pair["anon_questions"]):
                print(f"  A{i+1}: {anon}")
            print()


if __name__ == "__main__":
    main()
