# WindMon Tekercselés Kamera Monitor

Python GUI alkalmazás motor-tekercselési folyamat megfigyelésére USB vagy laptop kamerával. A program előképet mutat, egyetlen RGB pillanatképet rögzít, majd a rögzített képet egy külön feldolgozó ablakban jeleníti meg a hozzá tartozó becsült színspektrummal.

## Főbb funkciók

- USB kamera és laptop webkamera kezelése menüből vagy legördülőből.
- Előkép megjelenítése a választott kameráról.
- Egyetlen RGB kép rögzítése a pillanatnyi képkockából.
- Külön feldolgozó ablak a rögzített képhez és a spektrumhoz.
- Egérgörgős nagyítás a feldolgozó és a szűrt képen.
- Nagyított kép pásztázása jobb vagy középső egérgombbal.
- Pontok elhelyezése a rögzített képen.
- Pontonként külön színű, kontrasztos számjelölés a képen.
- Pontonként külön színű spektrumgörbe rajzolása a pont számának színével egyezően.
- Teljes képernyős spektrumnézet külön gombbal.
- Kattintható marker a spektrumgörbéken a hullámhossz és intenzitás leolvasásához.
- Teljes spektrumtartományos szűrés a markerrel kijelölt szélsőértékek alapján.
- Szűrési paraméterek mentése és betöltése JSON fájlba.
- Betöltött szűrősáv halvány szürke megjelenítése a spektrumon.
- A spektrum 90 hullámhossz-mintára van bontva 380 nm és 780 nm között.

## Fontos műszaki megjegyzés

Egy szokásos RGB kamera csak három színcsatornát mér, ezért a program nem valódi spektrométert valósít meg. A grafikon a kiválasztott pixel RGB értékei alapján becsült spektrumeloszlást mutat, ami vizuális összehasonlításra hasznos, de nem laboratóriumi mérés.

## Könyvtárak

- `opencv-python`
- `numpy`
- `matplotlib`
- `Pillow`
- `tkinter` a legtöbb Windows Python telepítésben alapból elérhető

## Telepítés

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Futtatás

```powershell
python app.py
```

## Használat

1. Indítsd el a programot.
2. Válaszd ki a kamerát a felső listából vagy a `Kamera` menüből.
3. Kattints a `Kép rögzítése` gombra.
4. A megnyíló feldolgozó ablakban kattints a rögzített képen a vizsgálni kívánt pontokra.
5. Egérgörgővel nagyíthatsz a képre, jobb vagy középső egérgombbal pedig mozgathatod a nagyított nézetet.
6. A spektrumdiagram ugyanebben az ablakban automatikusan frissül.
7. A `Spektrum teljes képernyőn` gombbal kinagyíthatod a spektrumot.
8. Kattints egy spektrumgörbére, hogy a marker kiírja a hullámhosszt és az intenzitást.
9. A `Szűrő mentése` gombbal JSON fájlba mentheted az aktuális markerpontokból számolt min-max spektrumsávot.
10. A `Szűrő betöltése` gombbal visszatölthetsz egy korábban mentett sávot; ez halvány szürke háttérként jelenik meg a spektrumon.
11. A `Szűrő alkalmazása` gombbal új ablakban jelenítheted meg azokat a pixeleket, amelyek minden hullámhosszon a kijelölt markerek minimuma és maximuma közé esnek. Ha nincs legalább két aktuális marker, de van betöltött szűrő, akkor a program a betöltött sávot használja.

## Projektfájlok

- `app.py` - teljes GUI alkalmazás
- `requirements.txt` - Python függőségek
- `README.md` - projektleírás

## Git

Lokális git verziózás indítása:

```powershell
git init
git add .
git commit -m "Initial camera GUI application"
```

GitHub feltöltéshez:

```powershell
git remote add origin <github-repo-url>
git branch -M main
git push -u origin main
```

Ha szeretnéd, a tároló URL-jével a GitHub remote beállítását is be tudom kötni.
