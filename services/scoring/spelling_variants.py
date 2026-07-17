"""UK -> US spelling equivalence table.

Used by:
  - WFDScorer  (services/scoring/rule_scorer.py) to normalise both the
    transcript and the user's text to a canonical US form before the
    multiset word-match — fixes the case where the transcript is US-spelled
    and the student legitimately writes British forms like "colour",
    "organise", "programme".
  - check_spelling (services/spelling_checker.py) to preprocess user text
    before pyspellchecker (US-only dict) — stops British variants from
    being flagged as candidate misspellings in the first place, so they
    never even reach the Claude judge fallback path.

Not used by FIBScorer (per product decision — kept scoped to WFD + the
writing spelling pipeline).

Includes base forms, plurals, past tense (-ed) and gerund (-ing) forms
because we normalise word-by-word — there is no morphological analyser
in the calling scorers.
"""

UK_TO_US: dict[str, str] = {
    # -our -> -or
    "colour": "color", "colours": "colors", "coloured": "colored",
    "colouring": "coloring", "colourful": "colorful",
    "favour": "favor", "favours": "favors", "favoured": "favored",
    "favouring": "favoring", "favourite": "favorite", "favourites": "favorites",
    "honour": "honor", "honours": "honors", "honoured": "honored",
    "honouring": "honoring", "honourable": "honorable",
    "labour": "labor", "labours": "labors", "laboured": "labored",
    "labouring": "laboring",
    "humour": "humor", "humours": "humors", "humoured": "humored",
    "humouring": "humoring", "humorous": "humorous",
    "neighbour": "neighbor", "neighbours": "neighbors",
    "neighbouring": "neighboring", "neighbourhood": "neighborhood",
    "neighbourhoods": "neighborhoods",
    "behaviour": "behavior", "behaviours": "behaviors",
    "behavioural": "behavioral",
    "endeavour": "endeavor", "endeavours": "endeavors",
    "endeavoured": "endeavored", "endeavouring": "endeavoring",
    "harbour": "harbor", "harbours": "harbors", "harboured": "harbored",
    "harbouring": "harboring",
    "rumour": "rumor", "rumours": "rumors", "rumoured": "rumored",
    "savour": "savor", "savours": "savors", "savoured": "savored",
    "savouring": "savoring",
    "splendour": "splendor",
    "vigour": "vigor",
    "flavour": "flavor", "flavours": "flavors", "flavoured": "flavored",
    "flavouring": "flavoring",
    "odour": "odor", "odours": "odors",
    "parlour": "parlor", "parlours": "parlors",
    "rigour": "rigor", "rigours": "rigors",

    # -re -> -er
    "centre": "center", "centres": "centers", "centred": "centered",
    "centring": "centering",
    "metre": "meter", "metres": "meters",
    "litre": "liter", "litres": "liters",
    "theatre": "theater", "theatres": "theaters",
    "fibre": "fiber", "fibres": "fibers",
    "calibre": "caliber", "calibres": "calibers",
    "sombre": "somber", "lustre": "luster", "spectre": "specter",
    "mitre": "miter",
    "kilometre": "kilometer", "kilometres": "kilometers",
    "millimetre": "millimeter", "millimetres": "millimeters",
    "centimetre": "centimeter", "centimetres": "centimeters",

    # -ise -> -ize
    "organise": "organize", "organises": "organizes",
    "organised": "organized", "organising": "organizing",
    "organisation": "organization", "organisations": "organizations",
    "organisational": "organizational",
    "realise": "realize", "realises": "realizes", "realised": "realized",
    "realising": "realizing", "realisation": "realization",
    "recognise": "recognize", "recognises": "recognizes",
    "recognised": "recognized", "recognising": "recognizing",
    "recognisable": "recognizable",
    "apologise": "apologize", "apologises": "apologizes",
    "apologised": "apologized", "apologising": "apologizing",
    "criticise": "criticize", "criticises": "criticizes",
    "criticised": "criticized", "criticising": "criticizing",
    "emphasise": "emphasize", "emphasises": "emphasizes",
    "emphasised": "emphasized", "emphasising": "emphasizing",
    "memorise": "memorize", "memorises": "memorizes",
    "memorised": "memorized", "memorising": "memorizing",
    "minimise": "minimize", "minimises": "minimizes",
    "minimised": "minimized", "minimising": "minimizing",
    "maximise": "maximize", "maximises": "maximizes",
    "maximised": "maximized", "maximising": "maximizing",
    "summarise": "summarize", "summarises": "summarizes",
    "summarised": "summarized", "summarising": "summarizing",
    "socialise": "socialize", "socialises": "socializes",
    "socialised": "socialized", "socialising": "socializing",
    "specialise": "specialize", "specialises": "specializes",
    "specialised": "specialized", "specialising": "specializing",
    "symbolise": "symbolize", "symbolises": "symbolizes",
    "symbolised": "symbolized", "symbolising": "symbolizing",
    "capitalise": "capitalize", "capitalises": "capitalizes",
    "capitalised": "capitalized", "capitalising": "capitalizing",
    "utilise": "utilize", "utilises": "utilizes",
    "utilised": "utilized", "utilising": "utilizing",
    "modernise": "modernize", "modernises": "modernizes",
    "modernised": "modernized", "modernising": "modernizing",
    "prioritise": "prioritize", "prioritises": "prioritizes",
    "prioritised": "prioritized", "prioritising": "prioritizing",
    "standardise": "standardize", "standardises": "standardizes",
    "standardised": "standardized", "standardising": "standardizing",
    "characterise": "characterize", "characterises": "characterizes",
    "characterised": "characterized", "characterising": "characterizing",
    "generalise": "generalize", "generalises": "generalizes",
    "generalised": "generalized", "generalising": "generalizing",
    "hospitalise": "hospitalize", "hospitalises": "hospitalizes",
    "hospitalised": "hospitalized", "hospitalising": "hospitalizing",
    "authorise": "authorize", "authorises": "authorizes",
    "authorised": "authorized", "authorising": "authorizing",
    "civilise": "civilize", "civilises": "civilizes",
    "civilised": "civilized", "civilising": "civilizing",
    "familiarise": "familiarize", "familiarises": "familiarizes",
    "familiarised": "familiarized", "familiarising": "familiarizing",
    "publicise": "publicize", "publicises": "publicizes",
    "publicised": "publicized", "publicising": "publicizing",
    "revolutionise": "revolutionize", "revolutionises": "revolutionizes",
    "revolutionised": "revolutionized", "revolutionising": "revolutionizing",
    "advertise": "advertise",  # NB: same in both dialects — kept for clarity

    # -yse -> -yze
    "analyse": "analyze", "analyses": "analyzes",
    "analysed": "analyzed", "analysing": "analyzing",
    "paralyse": "paralyze", "paralyses": "paralyzes",
    "paralysed": "paralyzed", "paralysing": "paralyzing",
    "catalyse": "catalyze", "catalyses": "catalyzes",
    "catalysed": "catalyzed",

    # -ogue -> -og
    "catalogue": "catalog", "catalogues": "catalogs",
    "catalogued": "cataloged",
    "dialogue": "dialog", "dialogues": "dialogs",
    "monologue": "monolog",
    "analogue": "analog", "analogues": "analogs",

    # -ence -> -ense
    "defence": "defense", "defences": "defenses",
    "defenceless": "defenseless",
    "offence": "offense", "offences": "offenses",
    "licence": "license", "licences": "licenses", "licenced": "licensed",
    "pretence": "pretense",

    # doubled -ll- (in inflections) -> single -l-
    "travelling": "traveling", "travelled": "traveled",
    "traveller": "traveler", "travellers": "travelers",
    "cancelling": "canceling", "cancelled": "canceled",
    "labelling": "labeling", "labelled": "labeled",
    "modelling": "modeling", "modelled": "modeled",
    "signalling": "signaling", "signalled": "signaled",
    "marvellous": "marvelous",
    "counselling": "counseling", "counselled": "counseled",
    "counsellor": "counselor", "counsellors": "counselors",
    "quarrelling": "quarreling", "quarrelled": "quarreled",
    "levelling": "leveling", "levelled": "leveled",
    "totalling": "totaling", "totalled": "totaled",

    # Irregulars
    "programme": "program", "programmes": "programs",
    "cheque": "check", "cheques": "checks",
    "plough": "plow", "ploughs": "plows", "ploughed": "plowed",
    "grey": "gray", "greys": "grays", "greying": "graying",
    "tyre": "tire", "tyres": "tires",
    "aluminium": "aluminum",
    "paediatric": "pediatric", "paediatrics": "pediatrics",
    "oestrogen": "estrogen",
    "mould": "mold", "moulds": "molds",
    "moulded": "molded", "moulding": "molding",
    "jewellery": "jewelry",
    "kerb": "curb", "kerbs": "curbs",
    "pyjamas": "pajamas",
    "storey": "story", "storeys": "stories",
    "sceptical": "skeptical", "scepticism": "skepticism",
    "sulphur": "sulfur", "sulphuric": "sulfuric",
    "manoeuvre": "maneuver", "manoeuvres": "maneuvers",
    "encyclopaedia": "encyclopedia",
    "artefact": "artifact", "artefacts": "artifacts",
    "draught": "draft", "draughts": "drafts",
    "aeroplane": "airplane", "aeroplanes": "airplanes",
    "practise": "practice", "practises": "practices",
    "practised": "practiced", "practising": "practicing",
    "enrol": "enroll", "enrols": "enrolls",
    "enrolment": "enrollment",
    "fulfil": "fulfill", "fulfils": "fulfills",
    "fulfilment": "fulfillment",
    "instil": "instill", "instils": "instills",
    "skilful": "skillful", "wilful": "willful",
    "ageing": "aging",
    "focussed": "focused", "focussing": "focusing",
    "biassed": "biased",
    "moustache": "mustache", "moustaches": "mustaches",
    "cosy": "cozy",
    "doughnut": "donut", "doughnuts": "donuts",
    "yoghurt": "yogurt", "yoghurts": "yogurts",
    "whilst": "while", "amongst": "among", "towards": "toward",
    "kilogramme": "kilogram", "gramme": "gram",
}


def to_us(word: str) -> str:
    """Return the canonical US spelling for ``word`` if a UK variant is known,
    otherwise the word unchanged. Case-insensitive lookup; returns lowercase.

    Callers are expected to have already lowercased and stripped punctuation.
    """
    return UK_TO_US.get(word, word)
