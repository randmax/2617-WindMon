# WindMon Tekercseles Kamera Monitor

Python GUI alkalmazas motor-tekercselesi folyamat megfigyelesere USB vagy laptop kameraval. A program elokepet mutat, egyetlen RGB pillanatkepet rogzit, majd a rogzitett kepet egy kulon feldolgozo ablakban jeleniti meg a hozzatartozo becsult szinspektrummal.

## Fobb funkciok

- USB kamera es laptop webkamera kezelese menubol vagy legordulobol.
- Elokep megjelenitese a valasztott kamerarol.
- Egyetlen RGB kep rogzitese a pillanatnyi kepkockabol.
- Kulon feldolgozo ablak a rogzitett kephez es a spektrumhoz.
- Pontok elhelyezese a rogzitett kepen.
- Pontonkent kulon szinu, kontrasztos szamjeloles a kepen.
- Pontonkent kulon szinu spektrumgorbe rajzolasa a pont szamanak szinevel egyezoen.
- Teljes kepernyos spektrum-nezet kulon gombbal.
- Kattinthato marker a spektrumgorbeken a hullamhossz es intenzitas leolvasasahoz.
- A spektrum 90 hullamhossz-mintara van bontva 380 nm es 780 nm kozott.

## Fontos muszaki megjegyzes

Egy szokasos RGB kamera csak harom szincsatornat mer, ezert a program nem valodi spektrometert valosit meg. A grafikon a kivalasztott pixel RGB ertekei alapjan becsult spektrumeloszlast mutat, ami vizualis osszehasonlitasra hasznos, de nem laboratoriumi meres.

## Konyvtarak

- `opencv-python`
- `numpy`
- `matplotlib`
- `Pillow`
- `tkinter` a legtobb Windows Python telepitesben alapbol elerheto

## Telepites

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Futtatas

```powershell
python app.py
```

## Hasznalat

1. Inditsd el a programot.
2. Valaszd ki a kamerat a felso listabol vagy a `Kamera` menubol.
3. Kattints a `Kep rogzitese` gombra.
4. A megnyilo feldolgozo ablakban kattints a rogzitett kepen a vizsgalni kivant pontokra.
5. A spektrum diagram ugyanebben az ablakban automatikusan frissul.
6. A `Spektrum teljes kepernyon` gombbal kinagyithatod a spektrumot.
7. Kattints egy spektrumgorbere, hogy a marker kiirja a hullamhosszt es az intenzitast.

## Projekt fajlok

- `app.py` - teljes GUI alkalmazas
- `requirements.txt` - Python fuggosegek
- `README.md` - projektleiras

## Git

Lokalis git verziozas inditasa:

```powershell
git init
git add .
git commit -m "Initial camera GUI application"
```

GitHub feltolteshez:

```powershell
git remote add origin <github-repo-url>
git branch -M main
git push -u origin main
```

Ha szeretned, a tarolo URL-jevel a GitHub remote beallitasat is be tudom kotni.
