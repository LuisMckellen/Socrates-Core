"""
synonyms.py — canonical-term expansion for Socratic key_terms matching.

Role in the architecture
------------------------
Used only on the key_terms side of ``state_machine._classify_by_key_terms``
(Change 4): a bank key term like ``"digest"`` should also match a student
answer that says "digesting". The student's answer is never expanded or
rewritten, only the bank's own vocabulary, so a synonym cannot introduce a
new correctness claim the bank did not already make.
"""

from __future__ import annotations

SYNONYM_MAP: dict[str, list[str]] = {
    "cells": ["cell"],
    "cell": ["cells"],
    "divide": ["split", "splits", "splitting", "separate", "separates"],
    "split": ["divide"],
    "splits": ["divide"],
    "splitting": ["divide"],
    "separate": ["divide", "separates", "separation", "apart"],
    "separates": ["divide", "separate"],
    "separation": ["separate"],
    "apart": ["separate"],
    "transcription": ["transcribe", "transcribes", "transcribing"],
    "transcribe": ["transcription"],
    "transcribes": ["transcription"],
    "transcribing": ["transcription"],
    "translation": ["translate", "translates", "translating"],
    "translate": ["translation"],
    "translates": ["translation"],
    "translating": ["translation"],
    "growth": ["grows", "growing", "grew"],
    "grows": ["growth"],
    "growing": ["growth"],
    "grew": ["growth"],
    "alive": ["living", "life", "survive", "survives", "survival"],
    "living": ["alive"],
    "life": ["alive"],
    "survive": ["alive"],
    "survives": ["alive"],
    "survival": ["alive"],
    "nucleus": ["nuclear"],
    "nuclear": ["nucleus"],
    "compartments": ["compartment", "compartmentalisation", "compartmentalization"],
    "compartment": ["compartments", "compartmentalization"],
    "compartmentalisation": ["compartments", "compartmentalization"],
    "compartmentalization": ["compartments", "compartmentalisation", "compartment"],
    "pre-existing": ["preexisting", "existing"],
    "preexisting": ["pre-existing"],
    "existing": ["pre-existing"],
    "digest": ["digesting", "digestion", "digests", "digestive"],
    "digesting": ["digest"],
    "digestion": ["digest"],
    "digests": ["digest"],
    "digestive": ["digest"],
    "gene"             : ["genes"],
    "substrate"        : ["substrates"],
    "spindle"          : ["spindle fiber", "spindle fibers", "microfilaments"],
    "replication"      : ["replicates", "replicated", "replicating", "replicate"],
    "mitosis"          : ["mitotic"],
    "chromatids"       : ["chromatid"],
    "distribution"     : ["distributions", "distributed", "distribute"],
    "heads"            : ["head"],
    "tails"            : ["tail"],
    "reduced"          : ["reduction", "reduce", "reduces"],
    "hydrocarbon"      : ["hydrocarbons", "hydrocarbon chain", "hydrocarbon chains"],
    "oxygen"           : ["oxygen atoms"],
    "active site"      : ["active-site", "active sites", "binding pocket", "binding site"],
    "activation energy": ["energy of activation"],
    "transition state" : ["transition states", "transition-state"],
    "selection"        : ["selective", "select", "selecting"],
    "diversity"        : ["diverse", "diversifies"],
    "population"       : ["populations"],
    "search"           : ["searches", "searching"],
}


def expand_terms(term: str) -> list[str]:
    """``term`` plus its known synonyms, ``term`` always first."""
    return [term, *SYNONYM_MAP.get(term.lower(), [])]
