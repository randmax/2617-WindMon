# WindMon Tekercseles Kamera Monitor

Python GUI alkalmazas motor-tekercselesi folyamat megfigyelesere USB vagy laptop kameraval. A program elokepet mutat, felvetelt keszit, RGB pillanatkepet rogzit, majd a rogzitett kepen elhelyezett pontokhoz becsult szinspektrumot rajzol.

## Fobb funkciok

- USB kamera es laptop webkamera kezelese menubol vagy legordulobol.
- Elokep megjelenitese a valasztott kamerarol.
- Video rogzitese `captures/` mappaba.
- RGB kep rogzitese a pillanatnyi kepkockabol.
- Pontok elhelyezese a rogzitett kepen.
- Pontonkent kulon szinu spektrumgorbe rajzolasa.
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
3. Szükség szerint inditsd el a felvetelt.
4. Kattints az `RGB kep rogzitese` gombra.
5. A jobb oldali rogzitett kepen kattints a vizsgalni kivant pontokra.
6. A spektrum diagram jobb oldalon automatikusan frissul.

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
