# Symulacja i analiza chaosu w układzie wahadła podwójnego w Pythonie.
Autor: Mateusz Pandura.
W kodzie wykorzystuna jest metoda Rungego–Kutty 4 rzędu do całkowania równań ruchu.
Zawartość to:
- animacja 1000 wahadeł podwójnych (z minimalnie różnymi warunkami początkowymi)
- porównanie dwóch trajektorii (wrażliwość na warunki początkowe)
- przestrzeń fazowa (θ₂, ω₂)
- analiza korelacji i autokorelacji
- wykresy powtarzalności (recurrence plots)
- analiza energii (kinetyczna, potencjalna, całkowita)
## Wymagania
- Python 3.x
- numpy
- matplotlib
- pandas
- seaborn
- scipy
- scikit-learn
- ffmpeg (zapis animacji)

Trzeba zwrócić uwagę na fakt, że symulacja 1000 wahadeł jest dość kosztowna obliczeniowo. W razie problemów z wydajnością, trzeba zmniejszyć ilość wahadeł
