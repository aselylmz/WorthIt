"""
Tarih yardımcı fonksiyonları.

Tarih dönüşümleri, ay/yıl hesaplamaları ve
zaman serisi indeksleme yardımcıları.
"""

from datetime import date, datetime
from typing import Tuple


def months_to_years_months(total_months: int) -> Tuple[int, int]:
    """Toplam ay sayısını yıl ve ay çiftine dönüştür.

    Parameters
    ----------
    total_months : int
        Toplam ay sayısı (>= 0).

    Returns
    -------
    tuple[int, int]
        (yıl, ay) çifti.

    Raises
    ------
    ValueError
        total_months negatif ise.

    Examples
    --------
    >>> months_to_years_months(100)
    (8, 4)
    >>> months_to_years_months(0)
    (0, 0)
    """
    if total_months < 0:
        raise ValueError(f"total_months negatif olamaz: {total_months}")
    return divmod(total_months, 12)


def add_months(start_date: date, months: int) -> date:
    """Bir tarihe belirtilen sayıda ay ekle.

    Parameters
    ----------
    start_date : date
        Başlangıç tarihi.
    months : int
        Eklenecek ay sayısı (>= 0).

    Returns
    -------
    date
        Sonuç tarih. Ay sonundaki taşma durumunda ayın
        son gününe yuvarlanır.

    Raises
    ------
    ValueError
        months negatif ise.

    Examples
    --------
    >>> from datetime import date
    >>> add_months(date(2025, 1, 31), 1)
    datetime.date(2025, 2, 28)
    """
    if months < 0:
        raise ValueError(f"months negatif olamaz: {months}")

    total_months = start_date.month - 1 + months
    new_year = start_date.year + total_months // 12
    new_month = total_months % 12 + 1

    # Ay sonu taşmasını ele al (ör: 31 Ocak + 1 ay → 28 Şubat)
    import calendar

    max_day = calendar.monthrange(new_year, new_month)[1]
    new_day = min(start_date.day, max_day)

    return date(new_year, new_month, new_day)


def format_duration_turkish(years: int, months: int) -> str:
    """Süreyi Türkçe formatla.

    Parameters
    ----------
    years : int
        Yıl sayısı.
    months : int
        Ay sayısı.

    Returns
    -------
    str
        Ör: "8 yıl 4 ay", "3 yıl", "7 ay".
    """
    parts = []
    if years > 0:
        parts.append(f"{years} yıl")
    if months > 0:
        parts.append(f"{months} ay")
    return " ".join(parts) if parts else "0 ay"
