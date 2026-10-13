"""Insect groups the ID step leaves out because they (practically) never land on a
light-trap screen.

The species list itself is used exactly as downloaded from GBIF (so its DOI
traces to the real file); these exclusions are applied on top, to BioCLIP's
Tree-of-Life labels, when the classifier's label set is built. Every ID written
with them active records ``EXCLUSIONS_ID`` so the result stays traceable.

Matching is on BioCLIP's own taxonomy (its order / family names), which is what
predictions come from. Keep the reasons honest — a group belongs here only if
an adult showing up on a screen *and* being recognisable at Mothbox resolution
is implausible. Groups whose winged adults do come to light (e.g. Strepsiptera
and Embioptera males, giant scales, alate ants and termites) are deliberately
NOT excluded.

Changing this table changes ``EXCLUSIONS_ID`` and so rebuilds the filtered
classifier cache on the next ID run.
"""

import hashlib

# (rank, name, reason). rank is "order" or "family" in BioCLIP's taxonomy.
LIGHT_TRAP_EXCLUSIONS = [
    # ── Parasites that live on their hosts ────────────────────────────────
    ("order", "Siphonaptera", "fleas — live on hosts and in nests"),
    ("order", "Phthiraptera", "lice — wingless, live their whole life on a host"),
    *[("family", fam, "lice (Psocodea) — wingless, live their whole life on a host") for fam in (
        "Pediculidae", "Pthiridae", "Haematopinidae", "Linognathidae", "Polyplacidae",
        "Hoplopleuridae", "Echinophthiriidae", "Enderleinellidae", "Pedicinidae",
        "Hamophthiriidae", "Haematomyzidae", "Menoponidae", "Boopidae", "Gyropidae",
        "Laemobothriidae", "Ricinidae", "Trimenoponidae", "Philopteridae", "Trichodectidae",
    )],
    ("family", "Cimicidae", "bed bugs and bat bugs — flightless, live in host roosts"),
    ("family", "Polyctenidae", "bat bugs — wingless ectoparasites"),
    ("family", "Nycteribiidae", "bat flies — wingless ectoparasites"),
    ("family", "Braulidae", "bee lice — wingless, live on honey bees"),
    ("family", "Arixeniidae", "bat earwigs — wingless, live in bat roosts"),
    ("family", "Hemimeridae", "rat earwigs — wingless, live on giant rats"),

    # ── Sessile scale insects ────────────────────────────────────────────
    # Females are fixed to plants (BioCLIP learned these from photos of them on
    # plants). Males of some species are winged but ~1 mm — not identifiable on a
    # screen photo. Giant scales (Margarodidae / Monophlebidae) and ensign scales
    # (Ortheziidae) are kept: their larger males do come to light.
    *[("family", fam, "scale insects — females sessile on plants; males ~1 mm") for fam in (
        "Diaspididae", "Coccidae", "Pseudococcidae", "Eriococcidae", "Dactylopiidae",
        "Kerriidae", "Asterolecaniidae", "Cerococcidae", "Lecanodiaspididae", "Kermesidae",
        "Aclerdidae", "Beesoniidae", "Conchaspididae", "Micrococcidae", "Phoenicococcidae",
        "Stictococcidae", "Halimococcidae", "Rhizoecidae", "Putoidae",
    )],

    # ── Wingless relict orders ────────────────────────────────────────────
    ("order", "Grylloblattodea", "ice crawlers — wingless, under rocks and ice at altitude"),
    ("order", "Notoptera", "ice crawlers (COL name) — wingless, under rocks and ice"),
    ("order", "Mantophasmatodea", "heelwalkers — wingless, southern African shrubland"),

    # ── Extinct (fossil-only) orders ──────────────────────────────────────
    # GBIF species lists include fossil specimens, and BioCLIP has labels for them.
    *[("order", order, "extinct — known only from fossils") for order in (
        "Palaeodictyoptera", "Protorthoptera", "Megasecoptera", "Diaphanopterodea",
        "Cnemidolestodea", "Cnemidolestida", "Meganisoptera", "Paoliida", "Caloneurodea",
        "Miomoptera", "Eoblattida", "Glosselytrodea", "Protocoleoptera", "Titanoptera",
        "Reculida", "Permopsocida", "Hypoperlida",
    )],
]

EXCLUDED_ORDERS = {name.lower() for rank, name, _ in LIGHT_TRAP_EXCLUSIONS if rank == "order"}
EXCLUDED_FAMILIES = {name.lower() for rank, name, _ in LIGHT_TRAP_EXCLUSIONS if rank == "family"}

# Short, stable identifier of this exact table — written on every ID made with it.
EXCLUSIONS_ID = "light-trap-exclusions@" + hashlib.sha256(
    repr(sorted((r, n.lower()) for r, n, _ in LIGHT_TRAP_EXCLUSIONS)).encode("utf-8")
).hexdigest()[:8]


def exclusion_reason(order, family):
    """Why a taxon with this BioCLIP order/family is excluded, or None if it isn't."""
    o, f = (order or "").lower(), (family or "").lower()
    for rank, name, reason in LIGHT_TRAP_EXCLUSIONS:
        if (rank == "order" and o == name.lower()) or (rank == "family" and f == name.lower()):
            return reason
    return None
