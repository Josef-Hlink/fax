# -*- coding: utf-8 -*-

from melee import Character


def peppi_character_to_internal(char_id: int) -> Character:
    """Peppi thinks it's funny to use a wildly different enum for characters..."""
    if char_id == 0x00:
        return Character.CPTFALCON
    if char_id == 0x01:
        return Character.DK
    if char_id == 0x02:
        return Character.FOX
    if char_id == 0x03:
        return Character.GAMEANDWATCH
    if char_id == 0x04:
        return Character.KIRBY
    if char_id == 0x05:
        return Character.BOWSER
    if char_id == 0x06:
        return Character.LINK
    if char_id == 0x07:
        return Character.LUIGI
    if char_id == 0x08:
        return Character.MARIO
    if char_id == 0x09:
        return Character.MARTH
    if char_id == 0x0A:
        return Character.MEWTWO
    if char_id == 0x0B:
        return Character.NESS
    if char_id == 0x0C:
        return Character.PEACH
    if char_id == 0x0D:
        return Character.PIKACHU
    if char_id == 0x0E:
        return Character.POPO  # Ice Climbers → Popo
    if char_id == 0x0F:
        return Character.JIGGLYPUFF
    if char_id == 0x10:
        return Character.SAMUS
    if char_id == 0x11:
        return Character.YOSHI
    if char_id == 0x12:
        return Character.ZELDA
    if char_id == 0x13:
        return Character.SHEIK
    if char_id == 0x14:
        return Character.FALCO
    if char_id == 0x15:
        return Character.YLINK
    if char_id == 0x16:
        return Character.DOC  # Dr. Mario → Doc
    if char_id == 0x17:
        return Character.ROY
    if char_id == 0x18:
        return Character.PICHU
    if char_id == 0x19:
        return Character.GANONDORF
    return Character.UNKNOWN_CHARACTER
