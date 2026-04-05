"""
Liste des actions cotées à la Bourse des Valeurs de Casablanca (BVC).
Les tickers Yahoo Finance pour les actions marocaines utilisent le suffixe .CS
"""

BVC_TICKERS = {
    # Banques & Finance
    "ATW": {"name": "Attijariwafa Bank", "secteur": "Banques", "yahoo": "ATW.CS"},
    "BCP": {"name": "Banque Centrale Populaire", "secteur": "Banques", "yahoo": "BCP.CS"},
    "BOA": {"name": "Bank Of Africa", "secteur": "Banques", "yahoo": "BOA.CS"},
    "CIH": {"name": "CIH Bank", "secteur": "Banques", "yahoo": "CIH.CS"},
    "CDM": {"name": "Crédit du Maroc", "secteur": "Banques", "yahoo": "CDM.CS"},
    "BMCE": {"name": "Bank Of Africa (ex-BMCE)", "secteur": "Banques", "yahoo": "BMCE.CS"},
    "HPS": {"name": "Hightech Payment Systems", "secteur": "Technologie financière", "yahoo": "HPS.CS"},
    "M2M": {"name": "M2M Group", "secteur": "Technologie financière", "yahoo": "M2M.CS"},
    "AFMA": {"name": "Afma", "secteur": "Finance", "yahoo": "AFMA.CS"},
    "AFRIC": {"name": "Africa", "secteur": "Assurances", "yahoo": "AFRIC.CS"},
    "WAFA": {"name": "Wafa Assurance", "secteur": "Assurances", "yahoo": "WAA.CS"},
    "MATU": {"name": "Atlanta", "secteur": "Assurances", "yahoo": "ATL.CS"},
    "SALAM": {"name": "Al Watanya", "secteur": "Assurances", "yahoo": "SAL.CS"},

    # Télécommunications
    "IAM": {"name": "Maroc Telecom (Itissalat Al-Maghrib)", "secteur": "Télécommunications", "yahoo": "IAM.CS"},

    # Immobilier
    "ADH": {"name": "Addoha", "secteur": "Immobilier", "yahoo": "ADH.CS"},
    "ALLT": {"name": "Alliances", "secteur": "Immobilier", "yahoo": "ALLT.CS"},
    "RES": {"name": "Résidences Dar Saada", "secteur": "Immobilier", "yahoo": "RDS.CS"},

    # Matériaux de construction & BTP
    "CMA": {"name": "Ciments du Maroc", "secteur": "Matériaux", "yahoo": "CMA.CS"},
    "LHM": {"name": "LafargeHolcim Maroc", "secteur": "Matériaux", "yahoo": "LHM.CS"},
    "HOLC": {"name": "Holcim Maroc", "secteur": "Matériaux", "yahoo": "HOLC.CS"},
    "SMI": {"name": "Société Métallurgique d'Imiter", "secteur": "Mines", "yahoo": "SMI.CS"},
    "MNG": {"name": "Managem", "secteur": "Mines", "yahoo": "MNG.CS"},

    # Distribution & Commerce
    "LBV": {"name": "Label' Vie", "secteur": "Distribution", "yahoo": "LBV.CS"},
    "MNEP": {"name": "Maghreb Oxygène", "secteur": "Distribution", "yahoo": "MNP.CS"},
    "AUTO": {"name": "Auto Nejma", "secteur": "Distribution", "yahoo": "AUTO.CS"},

    # Agroalimentaire
    "LES": {"name": "Lesieur Cristal", "secteur": "Agroalimentaire", "yahoo": "LES.CS"},
    "COSU": {"name": "Cosumar", "secteur": "Agroalimentaire", "yahoo": "CSR.CS"},
    "CBM": {"name": "Centrale Danone", "secteur": "Agroalimentaire", "yahoo": "CBM.CS"},
    "DARI": {"name": "Dari Couspate", "secteur": "Agroalimentaire", "yahoo": "DARI.CS"},

    # Énergie
    "TQM": {"name": "Taqa Morocco", "secteur": "Énergie", "yahoo": "TQM.CS"},
    "TMA": {"name": "Total Maroc", "secteur": "Énergie", "yahoo": "TMA.CS"},

    # Industrie
    "IBC": {"name": "Involys", "secteur": "Technologie", "yahoo": "IBC.CS"},
    "S2M": {"name": "S2M", "secteur": "Technologie", "yahoo": "S2M.CS"},
    "DLM": {"name": "Delattre Levivier Maroc", "secteur": "Industrie", "yahoo": "DLM.CS"},
    "STROC": {"name": "Stroc Industrie", "secteur": "Industrie", "yahoo": "STRO.CS"},

    # Transport & Logistique
    "CTM": {"name": "CTM", "secteur": "Transport", "yahoo": "CTM.CS"},
    "TIMAR": {"name": "Timar", "secteur": "Logistique", "yahoo": "TIM.CS"},

    # Tourisme & Hôtellerie
    "RIS": {"name": "Risma", "secteur": "Hôtellerie", "yahoo": "RIS.CS"},

    # Santé
    "PHAR": {"name": "Pharma 5", "secteur": "Santé", "yahoo": "PHR.CS"},

    # Indices
    "MASI": {"name": "MASI (Moroccan All Shares Index)", "secteur": "Indice", "yahoo": "^MASI"},
    "MADEX": {"name": "MADEX (Most Active Shares Index)", "secteur": "Indice", "yahoo": "^MADEX"},
}


def get_ticker_info(symbol: str) -> dict:
    """Retourne les informations d'un ticker BVC."""
    symbol = symbol.upper()
    if symbol in BVC_TICKERS:
        return BVC_TICKERS[symbol]
    # Recherche par ticker Yahoo
    for k, v in BVC_TICKERS.items():
        if v["yahoo"].upper() == symbol.upper():
            return {**v, "bvc_symbol": k}
    return {}


def get_tickers_by_sector(secteur: str) -> dict:
    """Retourne les tickers d'un secteur donné."""
    return {k: v for k, v in BVC_TICKERS.items() if v["secteur"].lower() == secteur.lower()}


def list_sectors() -> list:
    """Retourne la liste des secteurs disponibles."""
    return sorted(set(v["secteur"] for v in BVC_TICKERS.values()))
