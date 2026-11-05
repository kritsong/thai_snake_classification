"""Approved species-label mapping (author sign-off 7 October 2026).

Shared by the review-table builder and the full-species dataset builder.
label -> (new class or REMOVE, action, reason, zoologist note or "").
"""

ZW = "​"
REMOVE = None

# label -> (new class or REMOVE, action, reason, zoologist note or "")
MAPPING = {
    # Spelling, capitalisation and hidden characters
    "Ahaeltulla prasina": ("Ahaetulla prasina", "Fix spelling (merge)", "Misspelled genus", ""),
    "Ahealtulla prasina": ("Ahaetulla prasina", "Fix spelling (merge)", "Misspelled genus", ""),
    f"Boiga{ZW} cynodon": ("Boiga cynodon", "Fix hidden character (merge)", "Invisible zero-width character inside the name", ""),
    f"Chrysopelea{ZW} pelias": ("Chrysopelea pelias", "Fix hidden character", "Invisible zero-width character inside the name", ""),
    f"Enhydris{ZW} chanardi": ("Enhydris chanardi", "Fix hidden character (merge)", "Invisible zero-width character inside the name", ""),
    f"Pseudoxenodon macrops{ZW}": ("Pseudoxenodon macrops", "Fix hidden character", "Invisible zero-width character at the end of the name", ""),
    f"Rhabdophis{ZW} nigrocinctus": ("Rhabdophis nigrocinctus", "Fix hidden character", "Invisible zero-width character inside the name", ""),
    f"Trimeresurus{ZW} cardamom{ZW}ensis{ZW}": ("Trimeresurus cardamomensis", "Fix hidden character (merge)", "Invisible zero-width characters inside the name", ""),
    "Naja Kaouthia": ("Naja kaouthia", "Fix capitalisation (merge)", "Species epithet must be lower case", ""),
    "xenochrophis flavipunctatus": ("Fowlea flavipunctatus", "Fix capitalisation + synonym (merge)", "Lower-case genus; GBIF accepted name is Fowlea flavipunctatus", ""),
    "Rhabdophis chrysagos": ("Rhabdophis chrysargos", "Fix spelling (merge)", "GBIF variant match (95) to Rhabdophis chrysargos", ""),
    "Dendrelaphis coudolineatus": ("Dendrelaphis caudolineatus", "Fix spelling (merge)", "GBIF variant match (96) to Dendrelaphis caudolineatus", ""),
    "Sinomicrurus maclellandi": ("Sinomicrurus macclellandi", "Fix spelling (merge)", "GBIF variant match (98) to Sinomicrurus macclellandi", ""),
    "Calliophis bivirgata flaviceps": ("Calliophis bivirgatus", "Fix spelling + subspecies to species (merge)", "Variant spelling of Calliophis bivirgatus flaviceps", ""),
    "Elaphe taeniura yunanensis": ("Elaphe taeniura", "Fix spelling + subspecies to species (merge)", "Misspelled yunnanensis; species level", ""),
    # Synonyms and older names
    "Ptyas mucosus": ("Ptyas mucosa", "Merge synonym", "Older spelling; GBIF variant match (98) to Ptyas mucosa", ""),
    "Enhydris plumbea": ("Hypsiscopus plumbea", "Merge synonym", "GBIF: synonym of Hypsiscopus plumbea, which is also a local label", ""),
    "Dinodon septentrionalis": ("Lycodon septentrionalis", "Merge synonym", "GBIF: synonym of Lycodon septentrionalis, which is also a local label", ""),
    "Xenochrophis flavipunctatus": ("Fowlea flavipunctatus", "Rename synonym", "GBIF: synonym of Fowlea flavipunctatus",
        "Display name only (training unaffected): GBIF uses 'Fowlea flavipunctatus'; some checklists spell it 'Fowlea flavipunctata'. Choose the spelling for the paper."),
    "Orthriophis taeniurus ridleyi": ("Elaphe taeniura", "Merge synonym + subspecies to species", "GBIF: synonym of Elaphe taeniura ridleyi; species level", ""),
    "Bungarus wanghaotingi": ("Bungarus wanghaotingi", "Keep (check name)", "GBIF lists it as a synonym of Bungarus multicinctus; no B. multicinctus label exists locally",
        "Display name only (training unaffected): keep 'Bungarus wanghaotingi' or use 'Bungarus multicinctus'?"),
    # Subspecies to species
    "Calliophis maculiceps maculiceps": ("Calliophis maculiceps", "Subspecies to species (merge)", "Species level", ""),
    "Calliophis maculiceps smithi": ("Calliophis maculiceps", "Subspecies to species (merge)", "Species level", ""),
    "Calliophis bivirgatus flaviceps": ("Calliophis bivirgatus", "Subspecies to species", "Species level", ""),
    "Calliophis intestinalis lineata": ("Calliophis intestinalis", "Subspecies to species", "Species level", ""),
    "Oreocryptophis porphyraceus porphyraceus": ("Oreocryptophis porphyraceus", "Subspecies to species (merge)", "Species level", ""),
    "Oreocryptophis porphyraceus pulchra": ("Oreocryptophis porphyraceus", "Subspecies to species (merge)", "Species level", ""),
    "Ovophis monticola meridionalis": ("Ovophis monticola", "Subspecies to species (merge)", "Species level; merges with the 3 'Ovophis monticola' images",
        "Ovophis taxonomy in Thailand has been revised; confirm these images belong to O. monticola."),
    "Elaphe taeniura helfenbergeri": ("Elaphe taeniura", "Subspecies to species (merge)", "Species level", ""),
    "Elaphe taeniura ridleyi": ("Elaphe taeniura", "Subspecies to species (merge)", "Species level", ""),
    "Elaphe taeniura spp.": ("Elaphe taeniura", "Subspecies to species (merge)", "Species identified; subspecies unspecified", ""),
    "Pareas berdmorei berdmorei": ("Pareas berdmorei", "Subspecies to species", "Species level; kept separate from P. carinatus (author decision, 7 Oct 2026)", ""),
    "Trimeresurus sabahi fucatus": ("Trimeresurus sabahi", "Subspecies to species (merge)", "Species level; merges with the 23 'Trimeresurus sabahi' images",
        "Display name: Thai records are often treated as Trimeresurus fucatus. Keep 'T. sabahi' or rename the merged class 'T. fucatus'?"),
    # Genus-only and undescribed labels
    "Ahaetulla sp.": (REMOVE, "Remove (genus only)", "Not identified to species", ""),
    "boiga sp.": (REMOVE, "Remove (genus only)", "Not identified to species", ""),
    "Bungarus sp.": (REMOVE, "Remove (genus only)", "Not identified to species", ""),
    "Dendrelaphis sp.": (REMOVE, "Remove (genus only)", "Not identified to species", ""),
    "Homalopsis sp.": (REMOVE, "Remove (genus only)", "Not identified to species", ""),
    "Indotyphlops sp.": (REMOVE, "Remove (genus only)", "Not identified to species", ""),
    "Lycodon sp.": (REMOVE, "Remove (genus only)", "Not identified to species", ""),
    "Naja sp.": (REMOVE, "Remove (genus only)", "Not identified to species", ""),
    "Oligodon sp.": (REMOVE, "Remove (genus only)", "Not identified to species", ""),
    "Pareas sp.": (REMOVE, "Remove (genus only)", "Not identified to species", ""),
    "Ptyas sp.": (REMOVE, "Remove (genus only)", "Not identified to species", ""),
    "Python sp.": (REMOVE, "Remove (genus only)", "Not identified to species", ""),
    "Trimeresurus sp.": (REMOVE, "Remove (genus only)", "Not identified to species", ""),
    "Xenochrophis sp.": (REMOVE, "Remove (genus only)", "Not identified to species", ""),
    "Xenochrophis spp.": (REMOVE, "Remove (genus only)", "Not identified to species", ""),
    "Naja sp. nov.": (REMOVE, "Remove (undescribed species)", "Not a named species", ""),
    # Kept, but worth a zoologist's look
    "Xenochrophis piscator": ("Fowlea flavipunctatus", "Merge synonym (author decision)",
        "Thai snakes once called X. piscator are now usually assigned to Fowlea flavipunctatus; author approved the merge on 7 Oct 2026", ""),
    "Pareas malaccanus": ("Pareas malaccanus", "Keep (check name)", "GBIF found no species-level match", "Confirm this name is valid for the 4 images."),
}

# Expert revision (N. Ratnarathorn, 20 October 2026), applied to dataset v4 by
# build_taxonomy_revision_v4.py. MAPPING above stays as signed off, so the v3 and v4 builds
# reproduce. Bungarus wanghaotingi, Trimeresurus sabahi and Ovophis monticola were confirmed.
# label -> (final class, reason)
REVISION = {
    "Xenochrophis piscator": ("Fowlea piscator", "Distinct species from Fowlea flavipunctatus; reverses the merge of 7 Oct 2026"),
    "Pareas malaccanus": ("Asthenodipsas malaccana", "Current genus and name; merges with Asthenodipsas malaccanus"),
    "Asthenodipsas malaccanus": ("Asthenodipsas malaccana", "Current name (feminine ending)"),
}


def final_class(label):
    """Class of a label after MAPPING and REVISION (REMOVE for genus-level labels)."""
    if label in REVISION:
        return REVISION[label][0]
    return MAPPING[label][0] if label in MAPPING else label
