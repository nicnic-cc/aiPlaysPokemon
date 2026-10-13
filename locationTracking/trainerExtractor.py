"""
Trainer extractor for Pokemon FireRed / LeafGreen ROMs.

Reads every trainer NPC straight out of the ROM: which map it stands on, where,
which way it can see, and - the point of the exercise - its trainer id. Beating
trainer N sets story flag 0x500 + N (TRAINER_FLAGS_START in mgba_server.lua), so
the id is what lets the navigator drop a trainer from its destination list once
FLAGS says it has been beaten.

How it finds them:

  * gMapGroups -> each map header -> its object events. Every NPC is an
    ObjectEventTemplate (24 bytes) carrying a position, a trainer type, a sight
    range and a pointer to the script that runs when you talk to it.
  * That script is walked with a small Gen 3 script decoder, following goto /
    call / goto_if branches, until it reaches a `trainerbattle` command (opcode
    0x5C), which names the trainer id. An ordinary trainer's script *starts*
    with trainerbattle; gym leaders, the Elite Four and scripted grunts do some
    setup first (Brock's is 13 bytes in), which is why this decodes rather than
    pattern-matches - a byte search runs off the end of one script into its
    neighbour and reports the wrong trainer.
  * gTrainers supplies the name and class ("BUG CATCHER" + "RICK"), so labels
    come out the same as the ones already typed into tileData by hand.

What it does NOT find: battles started by stepping on a tile rather than talking
to someone (the Route 22 / Cerulean rival fights). Those have no NPC to place,
and mgba_server.lua's NAMED_FLAGS covers them.

Table addresses are located by scanning for their shape, so any FR/LG version
works; the scan is cached per version in KNOWN_ADDRS once verified.

CLI:
    python trainerExtractor.py                   # dump -> trainerData/romTrainers.json
    python trainerExtractor.py 1 0               # print trainers for bank 1, number 0
    python trainerExtractor.py --apply           # place trainers into tileData/*.json
    python trainerExtractor.py --apply --dry-run # report what --apply would change
    python trainerExtractor.py --rom <rom.gba> ...   # use a specific ROM

--apply needs each map's RAM -> image offset. A measured one in
connectionData/mapOffsets.json wins; otherwise it comes from the rip's size
against the ROM map's: the same size means (0, 0), 14x10 bigger means the
standard (7, 5) border. A rip framed any other way is skipped rather than
guessed - walk onto it once with the navigator running and it measures itself.

--apply is idempotent and never overwrites a hand-typed label or a category
other than "npc", so re-run it whenever the dump or mapOffsets.json changes.
"""

import json
import os
import struct
import sys

# ── Gen 3 character decode (matches mgba_server.lua GEN3_CHARS) ───────────────
_GEN3 = {0x00: " "}
for _i in range(26):
    _GEN3[0xBB + _i] = chr(65 + _i)   # A-Z
    _GEN3[0xD5 + _i] = chr(97 + _i)   # a-z
for _i in range(10):
    _GEN3[0xA1 + _i] = chr(48 + _i)   # 0-9
_GEN3.update({0xAE: "-", 0xB8: ",", 0xBA: "/", 0xAD: ".", 0xAB: "!", 0xAC: "?",
              0xB3: "'", 0xB4: "'", 0xB5: "M", 0xB6: "F", 0xB0: "...",
              0x2D: "&",    # "TWINS ELI & ANNE" - double-battle pairs
              0x1B: "E",    # e-acute, as in POKeMANIAC
              0x53: "PK", 0x54: "MN"})   # PKMN BREEDER / PKMN RANGER

ROM_BASE = 0x08000000
TRAINER_FLAGS_START = 0x500

# Verified table addresses, keyed by (game_code, version_byte). Anything missing
# is found by scanning; the scan prints what it found so it can be added here.
KNOWN_ADDRS = {
    ("BPGE", 1): {"trainers": 0x0823EB14, "mapGroups": 0x083526F8},  # LeafGreen v1.1
}

TRAINER_SIZE = 40          # struct Trainer
OBJECT_EVENT_SIZE = 24     # struct ObjectEventTemplate
MAP_GROUP_COUNT = 43       # MAP_GROUPS_COUNT in FR/LG
CLASS_NAME_LEN = 13        # gTrainerClassNames entry

# Default RAM -> image offset for interior rips; see locationTracker.BORDER_OFFSET.
BORDER_OFFSET = (7, 5)

# ── Gen 3 script decoding ─────────────────────────────────────────────────────
# Total instruction length (opcode included) for the commands that show up
# ahead of a trainerbattle. Anything not listed stops that branch: an unknown
# length means the next "opcode" would be a guess, and a guess is how you end up
# reading someone else's trainer id.
_FIXED_LEN = {
    0x00: 1, 0x01: 1, 0x09: 2, 0x0B: 3, 0x0E: 2, 0x0F: 6, 0x10: 3, 0x11: 6,
    0x12: 6, 0x13: 6, 0x14: 3, 0x15: 9, 0x16: 5, 0x17: 5, 0x18: 5, 0x19: 5,
    0x1A: 5, 0x1B: 3, 0x1C: 3, 0x1D: 6, 0x1E: 6, 0x1F: 6, 0x20: 9, 0x21: 5,
    0x22: 5, 0x23: 5, 0x25: 3, 0x26: 5, 0x27: 1, 0x28: 3, 0x29: 3, 0x2A: 3,
    0x2B: 3, 0x2F: 3, 0x30: 1, 0x31: 3, 0x32: 1, 0x33: 4, 0x34: 3, 0x35: 1,
    0x36: 3, 0x37: 2, 0x38: 2, 0x42: 5, 0x43: 1, 0x44: 5, 0x45: 5, 0x46: 5,
    0x47: 5, 0x48: 3, 0x4F: 7, 0x50: 9, 0x51: 3, 0x52: 5, 0x53: 3, 0x54: 5,
    0x55: 3, 0x56: 5, 0x57: 7, 0x58: 5, 0x59: 5, 0x5A: 1, 0x5B: 4, 0x5D: 1,
    0x60: 3, 0x61: 3, 0x62: 3, 0x63: 7, 0x64: 3, 0x65: 4, 0x66: 1, 0x67: 5,
    0x68: 1, 0x69: 1, 0x6A: 1, 0x6B: 1, 0x6C: 1, 0x6D: 1, 0x6E: 3, 0x6F: 5,
    0x70: 6, 0x71: 6, 0x72: 1, 0x75: 5, 0x76: 1, 0x7C: 3, 0x7D: 4, 0x7E: 2,
    0x7F: 4, 0x80: 4, 0x82: 4, 0x83: 4, 0x84: 4, 0x85: 6, 0x86: 5, 0x8F: 3,
    0x90: 6, 0x91: 6, 0x92: 6, 0x93: 4, 0x96: 3, 0x97: 2, 0x98: 3, 0x99: 3,
    0x9A: 2, 0x9B: 5, 0x9C: 3, 0x9D: 4, 0x9E: 3, 0x9F: 3, 0xA0: 1, 0xA1: 5,
    0xA2: 9, 0xA3: 1, 0xA4: 3, 0xA5: 1, 0xA6: 2, 0xA7: 3, 0xAC: 5, 0xAD: 5,
    0xAE: 1, 0xAF: 5, 0xB0: 5, 0xB3: 3, 0xB4: 3, 0xB5: 3, 0xB6: 6, 0xB7: 1,
    0xC4: 2, 0xC6: 1, 0xC7: 2,                                    # 0xC7 = FR textcolor
}
_END = {0x02, 0x03, 0x08, 0x0C, 0x0D, 0x24, 0x5E, 0x5F}   # control never falls through
_CALL, _GOTO, _GOTO_IF, _CALL_IF = 0x04, 0x05, 0x06, 0x07
_TRAINERBATTLE = 0x5C
# trainerbattle length by battle type: id + local id + 2..4 text/script pointers.
_BATTLE_LEN = {0: 14, 1: 18, 2: 18, 3: 10, 4: 18, 5: 14, 6: 22, 7: 18, 8: 22, 9: 14}
_MAX_SCRIPT_STEPS = 400


def _decode(data, start, maxLen=13):
    out, i = [], start
    for _ in range(maxLen):
        b = data[i]
        if b == 0xFF:
            return "".join(out)
        out.append(_GEN3.get(b, "?"))
        i += 1
    return None


class TrainerRom:
    def __init__(self, romPath):
        with open(romPath, 'rb') as f:
            self.data = f.read()
        self.gameCode = self.data[0xAC:0xB0].decode('ascii', 'replace')
        self.version = self.data[0xBC]
        known = KNOWN_ADDRS.get((self.gameCode, self.version), {})

        self.trainersAddr = known.get("trainers")
        if self.trainersAddr is None or self._trainerRun(self.trainersAddr) < 300:
            self.trainersAddr = self._scanTrainers()
        self.trainerCount = self._trainerRun(self.trainersAddr) + 1  # + dummy 0

        self.mapGroupsAddr = known.get("mapGroups")
        if self.mapGroupsAddr is None or not self._looksLikeMapGroups(self.mapGroupsAddr):
            self.mapGroupsAddr = self._scanMapGroups()

        self.classNamesAddr = self._findClassNames()

    # ── raw reads ──────────────────────────────────────────────────────────
    def u8(self, addr):
        return self.data[addr - ROM_BASE]

    def u16(self, addr):
        return struct.unpack_from("<H", self.data, addr - ROM_BASE)[0]

    def s16(self, addr):
        return struct.unpack_from("<h", self.data, addr - ROM_BASE)[0]

    def u32(self, addr):
        return struct.unpack_from("<I", self.data, addr - ROM_BASE)[0]

    def isPtr(self, v):
        return ROM_BASE <= v < ROM_BASE + len(self.data) - 4

    def text(self, addr):
        return _decode(self.data, addr - ROM_BASE)

    # ── table discovery ────────────────────────────────────────────────────
    def _trainerOk(self, addr):
        return (1 <= self.u8(addr + 0x20) <= 6 and self.isPtr(self.u32(addr + 0x24))
                and self.text(addr + 4) is not None)

    def _trainerRun(self, base):
        """How many valid trainers follow the dummy entry 0 at `base`."""
        n = 0
        while self.isPtr(base + TRAINER_SIZE * (n + 2)) and self._trainerOk(
                base + TRAINER_SIZE * (n + 1)):
            n += 1
        return n

    def _scanTrainers(self):
        for off in range(0, len(self.data) - TRAINER_SIZE * 300, 4):
            addr = ROM_BASE + off
            if self._trainerOk(addr + TRAINER_SIZE) and self._trainerRun(addr) >= 300:
                print(f"trainerExtractor: gTrainers found at 0x{addr:08X} "
                      f"(add to KNOWN_ADDRS)")
                return addr
        raise ValueError("Could not locate gTrainers in this ROM")

    def _headerOk(self, ptr):
        if not self.isPtr(ptr) or not self.isPtr(ptr + 0x1C):
            return False
        conns = self.u32(ptr + 12)
        return (self.isPtr(self.u32(ptr)) and self.isPtr(self.u32(ptr + 4))
                and self.isPtr(self.u32(ptr + 8)) and (conns == 0 or self.isPtr(conns)))

    def _groupOk(self, ptr):
        return (self.isPtr(ptr) and self._headerOk(self.u32(ptr))
                and self._headerOk(self.u32(ptr + 4)))

    def _looksLikeMapGroups(self, addr):
        return all(self._groupOk(self.u32(addr + 4 * i)) for i in range(40))

    def _scanMapGroups(self):
        for off in range(0, len(self.data) - 4 * MAP_GROUP_COUNT, 4):
            addr = ROM_BASE + off
            if self._groupOk(self.u32(addr)) and self._looksLikeMapGroups(addr):
                print(f"trainerExtractor: gMapGroups found at 0x{addr:08X} "
                      f"(add to KNOWN_ADDRS)")
                return addr
        raise ValueError("Could not locate gMapGroups in this ROM")

    def _findClassNames(self):
        """gTrainerClassNames, anchored on the classes of two known trainers.

        "BUG CATCHER" is in the ROM more than once, so one anchor isn't enough:
        each candidate base (hit minus RICK's class index) must also put
        "LEADER" at BROCK's class index.
        """
        anchors = {}
        for tid in range(1, self.trainerCount):
            name = self.trainerName(tid)
            if name in ("RICK", "BROCK") and name not in anchors:
                anchors[name] = self.u8(self.trainersAddr + TRAINER_SIZE * tid + 1)
        if len(anchors) < 2:
            return None
        needle = bytes(next(k for k, v in _GEN3.items() if v == ch)
                       for ch in "BUG CATCHER") + b"\xFF"
        start = 0
        while True:
            idx = self.data.find(needle, start)
            if idx < 0:
                return None
            base = ROM_BASE + idx - CLASS_NAME_LEN * anchors["RICK"]
            if self.isPtr(base) and \
                    self.text(base + CLASS_NAME_LEN * anchors["BROCK"]) == "LEADER":
                return base
            start = idx + 1

    # ── trainers ───────────────────────────────────────────────────────────
    def trainerName(self, tid):
        return self.text(self.trainersAddr + TRAINER_SIZE * tid + 4) or f"#{tid}"

    def trainerClass(self, tid):
        cls = self.u8(self.trainersAddr + TRAINER_SIZE * tid + 1)
        if self.classNamesAddr is None:
            return cls, ""
        addr = self.classNamesAddr + CLASS_NAME_LEN * cls
        name = self.text(addr) or ""
        # "SWIMMER" + the male/female symbol would decode as "SWIMMERM".
        if name and self.u8(addr + len(name) - 1) in (0xB5, 0xB6):
            name = name[:-1]
        return cls, name

    def battlesInScript(self, scriptPtr):
        """[(trainerId, battleType)] for every trainerbattle the script reaches.

        Walks branches breadth-first and stops a branch at the first command it
        can't size. In practice an object's script names one trainer; several
        distinct ids mean the same NPC fights differently depending on state,
        and any one of them being beaten counts.
        """
        found, seen, queue = [], set(), [scriptPtr]
        steps = 0
        while queue and steps < _MAX_SCRIPT_STEPS:
            pc = queue.pop(0)
            while self.isPtr(pc) and pc not in seen and steps < _MAX_SCRIPT_STEPS:
                seen.add(pc)
                steps += 1
                op = self.u8(pc)
                if op == _TRAINERBATTLE:
                    btype, tid = self.u8(pc + 1), self.u16(pc + 2)
                    if btype not in _BATTLE_LEN or not 1 <= tid < self.trainerCount:
                        break
                    if (tid, btype) not in found:
                        found.append((tid, btype))
                    pc += _BATTLE_LEN[btype]
                elif op in _END:
                    break
                elif op in (_CALL, _GOTO):
                    target = self.u32(pc + 1)
                    if self.isPtr(target):
                        queue.append(target)
                    if op == _GOTO:
                        break
                    pc += 5
                elif op in (_GOTO_IF, _CALL_IF):
                    target = self.u32(pc + 2)
                    if self.isPtr(target):
                        queue.append(target)
                    pc += 6
                elif op in _FIXED_LEN:
                    pc += _FIXED_LEN[op]
                else:
                    break
        return found

    # ── maps ───────────────────────────────────────────────────────────────
    def _maps(self):
        """Yield (bank, number, headerPtr) for every map in gMapGroups.

        Groups sit back to back in ROM with no terminator, so a group ends where
        the next group's table begins.
        """
        groups = [self.u32(self.mapGroupsAddr + 4 * i) for i in range(MAP_GROUP_COUNT)]
        # The group table itself follows the last group's list, so it's an end too.
        starts = set(groups) | {self.mapGroupsAddr}
        for bank, gptr in enumerate(groups):
            if not self.isPtr(gptr):
                continue
            num = 0
            while num < 256:
                slot = gptr + 4 * num
                if num > 0 and slot in starts:
                    break
                header = self.u32(slot)
                if not self._headerOk(header):
                    break
                yield bank, num, header
                num += 1

    def extractAll(self):
        """{"bank,number": {width, height, trainers: [...]}} for maps with trainers.

        width/height are the map's size in the ROM, which is what lets --apply
        work out how a rip is framed without anyone measuring it.
        """
        out = {}
        for bank, num, header in self._maps():
            layout = self.u32(header)
            events = self.u32(header + 4)
            count, objs = self.u8(events), self.u32(events + 4)
            if not count or not self.isPtr(objs):
                continue
            trainers = []
            for i in range(count):
                o = objs + OBJECT_EVENT_SIZE * i
                script = self.u32(o + 0x10)
                if not self.isPtr(script):
                    continue
                battles = self.battlesInScript(script)
                if not battles:
                    continue
                ids = []
                for tid, _btype in battles:
                    if tid not in ids:
                        ids.append(tid)
                tid = ids[0]
                cls, className = self.trainerClass(tid)
                name = self.trainerName(tid)
                trainers.append({
                    "localId": self.u8(o),
                    "x": self.s16(o + 4),
                    "y": self.s16(o + 6),
                    "trainerIds": ids,
                    "flags": [f"0x{TRAINER_FLAGS_START + t:X}" for t in ids],
                    "name": name,
                    "trainerClass": cls,
                    "className": className,
                    "label": f"{className} {name}".strip(),
                    # Type 0 = only fights when spoken to (gym leaders, bosses);
                    # otherwise it walks up to you once inside sightRange.
                    "trainerType": self.u8(o + 0x0C),
                    "sightRange": self.u16(o + 0x0E) if self.u8(o + 0x0C) else 0,
                    "battleType": battles[0][1],
                    # Event flag that hides this NPC once set (0 = never hidden).
                    "hideFlag": f"0x{self.u16(o + 0x14):X}",
                })
            if trainers:
                out[f"{bank},{num}"] = {
                    "width": self.u32(layout),
                    "height": self.u32(layout + 4),
                    "trainers": trainers,
                }
        return out


# ── applying to tileData ──────────────────────────────────────────────────────
# Tile types a trainer may stand on: floor in all its painted forms (cave floor
# is painted tall_grass so encounters work there; swimmers stand in water).
# Anything else under a trainer - a door, a warp, a ledge - means the offset is
# wrong, not the map.
_OBJECT_TYPE = 14
_REPLACEABLE_TILES = {0, 1, 2, 3, 4, _OBJECT_TYPE}
TRAINER_CATEGORY = "trainer"

# How rips relate to the ROM map. Checked against every painted map by lining
# the ROM's collision grid up with the painted one: every measured offset in
# mapOffsets.json reproduces, interiors all sit at +14x10 with (7, 5), and the
# overworld rips are the bare map at (0, 0) - except Pallet Town, which is why
# a measured offset still wins.
_SAME_SIZE_OFFSET = (0, 0)
_PADDED = (2 * BORDER_OFFSET[0], 2 * BORDER_OFFSET[1])


def _here(*parts):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


def romKeyForMap(mapName):
    """'1-0-ViridianForest' -> '1,0' (map files are named bank-number-Name)."""
    parts = mapName.split('-', 2)
    if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
        return f"{int(parts[0])},{int(parts[1])}"
    return None


def loadMapOffsets():
    path = _here('connectionData', 'mapOffsets.json')
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        raw = json.load(f)
    return {k: tuple(v) for k, v in raw.items()
            if not k.startswith('_') and isinstance(v, list) and len(v) == 2}


def offsetForMap(mapName, mapOffsets, imageSize, romSize):
    """RAM -> image offset, or None when it would be a guess.

    A measured offset wins; otherwise the rip's size relative to the ROM map
    decides it. A rip that is neither the bare map nor the standard padded one
    is framed some other way, and is skipped until the navigator measures it.
    """
    if mapName in mapOffsets:
        return mapOffsets[mapName]
    dw, dh = imageSize[0] - romSize[0], imageSize[1] - romSize[1]
    if (dw, dh) == (0, 0):
        return _SAME_SIZE_OFFSET
    if (dw, dh) == _PADDED:
        return BORDER_OFFSET
    return None


def placeTrainers(data, trainers, offset):
    """Merge ROM trainers into one tileData dict in place.

    Returns a list of human-readable change/warning lines. Existing labels win
    over the ROM's ("Brock" stays "Brock"), and so does any category other than
    plain "npc" - a gym leader filed under "gym" keeps it - but every trainer
    gets its ids in objectTrainers, which is what the navigator prunes on.
    """
    notes = []
    grid = data.get('tiles') or []
    height, width = len(grid), len(grid[0]) if grid else 0
    objects = data.setdefault('objects', {})
    cats = data.setdefault('objectCategories', {})
    trainerIds = data.setdefault('objectTrainers', {})
    dx, dy = offset

    for t in trainers:
        col, row = t['x'] + dx, t['y'] + dy
        key = f"{row},{col}"
        label = t['label']
        if not (0 <= row < height and 0 <= col < width):
            notes.append(f"  skip {label}: ({col},{row}) is off the {width}x{height} map")
            continue
        tile = grid[row][col]
        if tile not in _REPLACEABLE_TILES:
            notes.append(f"  skip {label}: ({col},{row}) is tile type {tile}, "
                         f"not floor - offset {list(offset)} looks wrong")
            continue

        changed = []
        if tile != _OBJECT_TYPE:
            grid[row][col] = _OBJECT_TYPE
            changed.append("tile")
        if key not in objects:
            objects[key] = label
            changed.append("label")
        if cats.get(key) in (None, "npc"):
            if cats.get(key) != TRAINER_CATEGORY:
                cats[key] = TRAINER_CATEGORY
                changed.append("category")
        if trainerIds.get(key) != t['trainerIds']:
            trainerIds[key] = list(t['trainerIds'])
            changed.append("ids")
        if changed:
            notes.append(f"  {objects[key]!r} at ({col},{row}) [{cats[key]}] "
                         f"ids={t['trainerIds']}: set {', '.join(changed)}")

    # Hand-placed npc labels a tile away from a trainer are almost certainly the
    # same person clicked slightly off. Flag rather than delete.
    placed = {tuple(int(v) for v in k.split(',')) for k in trainerIds}
    for key, cat in cats.items():
        if cat != "npc":
            continue
        r, c = (int(v) for v in key.split(','))
        if any(abs(r - pr) + abs(c - pc) == 1 for pr, pc in placed):
            notes.append(f"  check npc {objects.get(key)!r} at ({c},{r}): sits next "
                         f"to a ROM trainer, possibly a duplicate")
    return notes


def placeTrainersOnMap(mapName, data, romTrainers, mapOffsets=None):
    """Place one map's ROM trainers into its tileData dict.

    Returns (notes, None) on success, or (None, reason) when the map can't be
    placed. Shared by --apply and mapEditor's "Import trainers" button.
    """
    entry = romTrainers.get(romKeyForMap(mapName) or '')
    if not entry or not entry.get('trainers'):
        return None, "no trainers in the ROM for this map"
    if not data.get('tiles'):
        return None, "no tile grid yet"
    if mapOffsets is None:
        mapOffsets = loadMapOffsets()
    imageSize = (len(data['tiles'][0]), len(data['tiles']))
    offset = offsetForMap(mapName, mapOffsets, imageSize, (entry['width'], entry['height']))
    if offset is None:
        return None, (f"rip is {imageSize[0]}x{imageSize[1]} vs ROM "
                      f"{entry['width']}x{entry['height']} - framing unknown; let the "
                      f"navigator measure it into mapOffsets.json first")
    notes = placeTrainers(data, entry['trainers'], offset)
    return [f"offset {list(offset)}"] + notes, None


def applyToTileData(romTrainers, dryRun=False):
    tileDir = _here('tileData')
    mapOffsets = loadMapOffsets()
    placedMaps, skipped = 0, []
    for fname in sorted(os.listdir(tileDir)):
        if not fname.endswith('.json'):
            continue
        mapName = fname[:-5]
        if romKeyForMap(mapName) not in romTrainers:
            continue
        path = os.path.join(tileDir, fname)
        with open(path, 'r') as f:
            data = json.load(f)
        notes, reason = placeTrainersOnMap(mapName, data, romTrainers, mapOffsets)
        if reason:
            skipped.append(f"{mapName}: {reason}")
            continue
        if len(notes) > 1:
            print(f"{mapName}  {notes[0]}")
            print("\n".join(notes[1:]))
            placedMaps += 1
            if not dryRun:
                with open(path, 'w') as f:
                    json.dump(data, f, separators=(',', ':'))
    print(f"\n{'Would update' if dryRun else 'Updated'} {placedMaps} map(s).")
    if skipped:
        print("Skipped:")
        for s in skipped:
            print(f"  {s}")


def loadRomTrainers():
    """The dumped table, {} if trainerExtractor hasn't been run."""
    path = _here('trainerData', 'romTrainers.json')
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _defaultRom():
    mgbaDir = _here('..', 'mGBA')
    for f in os.listdir(mgbaDir):
        if f.lower().endswith('.gba'):
            return os.path.join(mgbaDir, f)
    raise FileNotFoundError("No .gba ROM found in mGBA/")


def main():
    args = sys.argv[1:]
    romPath = None
    if '--rom' in args:
        i = args.index('--rom')
        romPath = args[i + 1]
        del args[i:i + 2]
    apply, dryRun = '--apply' in args, '--dry-run' in args
    args = [a for a in args if a not in ('--apply', '--dry-run')]

    rom = TrainerRom(romPath or _defaultRom())
    print(f"ROM {rom.gameCode} v{rom.version}  gTrainers=0x{rom.trainersAddr:08X} "
          f"({rom.trainerCount})  gMapGroups=0x{rom.mapGroupsAddr:08X}  "
          f"classNames={'0x%08X' % rom.classNamesAddr if rom.classNamesAddr else 'not found'}")
    allTrainers = rom.extractAll()

    if len(args) >= 2:
        print(json.dumps(allTrainers.get(f"{args[0]},{args[1]}", []), indent=2))
        return

    outDir = _here('trainerData')
    os.makedirs(outDir, exist_ok=True)
    outPath = os.path.join(outDir, 'romTrainers.json')
    with open(outPath, 'w') as f:
        json.dump(allTrainers, f, indent=1)
    total = sum(len(v['trainers']) for v in allTrainers.values())
    print(f"Dumped {total} trainer NPCs on {len(allTrainers)} maps to {outPath}")

    if apply:
        print()
        applyToTileData(allTrainers, dryRun=dryRun)


if __name__ == '__main__':
    main()
