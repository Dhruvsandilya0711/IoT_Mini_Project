"""Label mappings for CICIoT2023: 34 raw classes -> 8 categories -> 2 classes.

The 33 attacks are grouped into the 7 categories of Neto et al. (2023);
together with benign traffic that gives the 8-class task.
"""

import pandas as pd

LABEL_COL = "label"

# Raw class name -> 8-class category.
CATEGORY_OF = {
    # DDoS (12)
    "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-HTTP_Flood": "DDoS",
    "DDoS-ICMP_Flood": "DDoS",
    "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-PSHACK_Flood": "DDoS",
    "DDoS-RSTFINFlood": "DDoS",
    "DDoS-SYN_Flood": "DDoS",
    "DDoS-SlowLoris": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS",
    "DDoS-TCP_Flood": "DDoS",
    "DDoS-UDP_Flood": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS",
    # DoS (4)
    "DoS-HTTP_Flood": "DoS",
    "DoS-SYN_Flood": "DoS",
    "DoS-TCP_Flood": "DoS",
    "DoS-UDP_Flood": "DoS",
    # Mirai (3)
    "Mirai-greeth_flood": "Mirai",
    "Mirai-greip_flood": "Mirai",
    "Mirai-udpplain": "Mirai",
    # Reconnaissance (5)
    "Recon-HostDiscovery": "Recon",
    "Recon-OSScan": "Recon",
    "Recon-PingSweep": "Recon",
    "Recon-PortScan": "Recon",
    "VulnerabilityScan": "Recon",
    # Spoofing (2)
    "DNS_Spoofing": "Spoofing",
    "MITM-ArpSpoofing": "Spoofing",
    # Brute force (1)
    "DictionaryBruteForce": "BruteForce",
    # Web-based (6)
    "Backdoor_Malware": "Web",
    "BrowserHijacking": "Web",
    "CommandInjection": "Web",
    "SqlInjection": "Web",
    "Uploading_Attack": "Web",
    "XSS": "Web",
    # Benign
    "BenignTraffic": "Benign",
}

# Display order, largest class first (by share of the full dataset).
CATEGORIES_8 = ["DDoS", "DoS", "Mirai", "Benign", "Spoofing", "Recon", "Web", "BruteForce"]
CLASSES_2 = ["Attack", "Benign"]
CLASSES_34 = [raw for cat in CATEGORIES_8 for raw in sorted(CATEGORY_OF) if CATEGORY_OF[raw] == cat]

LEVELS = (2, 8, 34)

_CANONICAL = {name.lower(): name for name in CATEGORY_OF}


def canonical(raw_labels: pd.Series) -> pd.Series:
    """Normalise raw label strings (case, whitespace) to the names in CATEGORY_OF."""
    uniques = pd.Series(raw_labels.unique())
    mapping = {u: _CANONICAL.get(str(u).strip().lower()) for u in uniques}
    unknown = sorted(str(u) for u, name in mapping.items() if name is None)
    if unknown:
        raise ValueError(f"Unknown CICIoT2023 labels: {unknown}. Add them to CATEGORY_OF in src/labels.py.")
    return raw_labels.map(mapping)


def to_level(raw_labels: pd.Series, level: int) -> pd.Series:
    """Map canonical 34-class labels to the requested granularity (2, 8 or 34)."""
    if level == 34:
        return raw_labels
    categories = raw_labels.map(CATEGORY_OF)
    if level == 8:
        return categories
    if level == 2:
        return categories.where(categories == "Benign", "Attack")
    raise ValueError(f"level must be one of {LEVELS}, got {level}")


def classes_for(level: int) -> list[str]:
    """All class names for a level, in display order."""
    return {2: CLASSES_2, 8: CATEGORIES_8, 34: CLASSES_34}[level]


def category_members(name: str) -> list[str]:
    """Raw labels belonging to a category name, or [name] if it is already a raw label."""
    members = [raw for raw, cat in CATEGORY_OF.items() if cat.lower() == name.lower()]
    if members:
        return members
    return canonical(pd.Series([name])).tolist()
