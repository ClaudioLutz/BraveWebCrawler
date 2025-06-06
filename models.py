# ─── models.py ────────────────────────────────────────────────────────────────
from dataclasses import dataclass, asdict, fields

@dataclass
class CompanyFacts:
    official_website: str | None = None
    founded: str | None = None
    haupt_sitz: str | None = None          # Official, from Zefix if possible
    firmen_id: str | None = None           # CHE-xxx.xxx.xxx
    haupt_tel: str | None = None
    haupt_mail: str | None = None
    geschaeftsbericht: str | None = None   # URL/PDF

    def missing_fields(self) -> list[str]:
        """Returns a list of field names that are still None or empty."""
        return [f.name for f in fields(self) if not getattr(self, f.name)]

    def merge_with(self, other: 'CompanyFacts'):
        """Merges fields from another CompanyFacts instance into this one, without overwriting existing data."""
        for f in fields(self):
            if not getattr(self, f.name) and getattr(other, f.name):
                setattr(self, f.name, getattr(other, f.name))
