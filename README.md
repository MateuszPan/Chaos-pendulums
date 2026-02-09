# Chaos-pendulums
symulacja i analiza dynamiki wahadła podwójnego w Pythonie.
Kod wykorzystuje metodę Rungego–Kutty 4 rzędu do całkowania równań ruchu.
Zawartość
- animacja 1000 wahadeł podwójnych z minimalnie różnymi warunkami początkowymi
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

Mała uwaga: Symulacja 1000 wahadeł jest kosztowna obliczeniowo. W razie problemów z wydajnością należy zmniejszyć liczbę wahadeł.
