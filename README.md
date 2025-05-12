Választott projekt:
7. Gradiens-módszer 
Gradiens módszer változatok összehasonlítása numpy-ban, költségfelület vizualizációval
- Különböző gradiens-módszer változatok ("Vanilla" SGD, Momentum, Nesterov, AdaGrad, AdaDelta, RMSprop, Adam, Nadam, stb. közül néhány) implementációja Numpy-ban, vagy meglévő implementációk felhasználása. 
- Szabálytalan alakú költség felületek generálása (két változóra) és a különböző algoritmusok iterálásának vizualizációja a felületen. (elég plot-okat vagy videókat gyártani pl. matplotlib-bel/OpenCV-vel)

# PyTorch Optimalizáció Vizualizáló

A projekt különböző "gradiens descent" optimalizálási algoritmusokat implementál és vizualizál PyTorch segítségével. Lehetővé teszi a különböző optimalizálók teljesítményének összehasonlítását pár 1D és 2D függvényen, és a hiperparaméter-hangolás hatásra is mutat példát.


## Leírás

A projektben megtötétnt több optimalizálási algoritmus implementálása, egy alap `Optimiser` absztrakt osztályból örökölve. Ezeket az optimalizáló függvényeket ezután különböző matematikai függvények minimalizálására használom, és lépéseiket 3D térben vizualizálom a Plotly segítségével. A projekt tartalmaz példát az optimalizálási folyamat animálására és a változó hiperparaméterek, mint például a tanulási ráta és a momentum együtthatók hatásainak tesztelésére.

## Features

### Implemented Optimizers:

* **Vanilla SGD**: Basic Stochastic Gradient Descent.
* **SignSGD**: SGD variant using the sign of the gradient.
* **Momentum SGD**: Incorporates momentum to accelerate convergence.
* **Nesterov Accelerated Gradient (NAG)**: A modification of momentum SGD with improved convergence properties.
* **Quasi-Hyperbolic Momentum (QHM)**: A weighted average of momentum and plain SGD.
* **RMSprop**: Uses a moving average of squared gradients to adapt the learning rate.
* **AdaGrad**: Adapts the learning rate based on the historical sum of squared gradients.
* **AdaDelta**: An extension of AdaGrad that seeks to reduce its aggressive, monotonically decreasing learning rate.
* **Adam**: Combines ideas from RMSprop and Momentum.
* **Nadam**: Adam incorporating Nesterov momentum.
* **AMSGrad**: A variant of Adam aiming to fix convergence issues.
* **Barzilai-Borwein (BB)**: A two-point step size gradient method.
* **Muon**: Uses the Newton-Schulz iteration for matrix inversion approximation within the optimization step (based on arXiv:2502.16982v1).

https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Extensions_and_variants

### Vizualizáció:

* **Statikus 3D Ábrák**: Plotly-t használok interaktív 3D felületi ábrák generálására, amelyek megmutatják az optimalizálandó fv. felületét és a különböző optimalizálók által bejárt utakat.
* **Animált 3D Ábrák**: Animációk (Plotly-vel), amelyek lépésről lépésre mutatják be az egyes optimalizálók haladását (általában a minimum felé).

### Teszt Függvények:

Az optimalizálókat különféle függvényeken tesztelem:
* Egyszerű kvadratikus függvények ($f(x) = x^2$, $f(x,y) = x^2 + y^2/2$)
* Oszcillációkat tartalmazó függvények ($f(x) = x^2 + \sin(10x)$)
* Himmelblau-függvény
* Lokális minimumokkal és nyeregpontokkal rendelkező függvények

forrás: https://en.wikipedia.org/wiki/Test_functions_for_optimization

## Függőségek

* Python 3
* PyTorch
* NumPy
* Pandas
* Matplotlib (for 1D plots)
* Plotly (for 3D plots and animations)

## Használat

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/matprog20250428/matprog
    cd matprog
    ```
2.  **Install dependencies:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install torch numpy pandas matplotlib plotly
    ```
3.  **Jupyter Notebook futtatása:**
    Órán tanultak szerint, pl VS Code vagy PyCharm etc.

## Notebook blokk-struktúra

* **Importok és Beállítás**: Importálja a szükséges könyvtárakat és beállítja a gyorsítókártya-használatot (ha elérhető) (CPU/GPU).
* **Optimalizáló Osztályok**: Definiálja az alap `Optimiser` osztályt és a különböző algoritmusok implementációit.
* **1-D Példák**: Egyszerű példák, amelyek bemutatják az optimalizálók viselkedését 1D függvényeken Matplotlib segítségével.
* **Optimalizációs Ciklus**: Tartalmazza a `run_optimization` függvényt az optimalizálási folyamat végrehajtásához egy adott optimalizálóra és függvényre.
* **Több Optimalizáló Futtatása**: Definiál egy szótárat az optimalizálókról és a `run_optimizations` függvényt több algoritmus futtatásához és eredményeinek gyűjtéséhez.
* **Vizualizációs Függvények**: Tartalmazza a `plot` és `animate` függvényeket (Plotly a 3D vizualizációhoz)
* **Hiperparaméter Kísérletek**: Különböző tanulási ráták (lr) és momentum értékek tesztelésére pár próbálkozás.
* **Tesztelés Más Felületeken**: Alkalmazza az optimalizálókat komplexebb függvényekre, mint a Himmelblau-függvény és nyeregpontokkal vagy több lokális minimummal rendelkező függvények.
