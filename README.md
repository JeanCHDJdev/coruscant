# coruscant 
---------------------------------------------------------

"There are people from a thousand worlds on Coruscant. Worlds that have faced similar struggles. You may see it as a city of metal, but there is wisdom to be found here. And allies." - *Qui-Gon Jinn*

### A toolkit package for astronomical and cosmological analysis

-----------------------------------------------------------------

Coruscant, the world city, the ecumenopolis, capital of the Old Republic. And also my personal place to store code I find useful for cosmological pipelines, analysis, papers, plots, and more. To be built as I go !


### Installation: 
------------------

Coruscant uses `pixi` for environment management and editable installation.
To create the environment and register the default Jupyter kernel, run
```bash
pixi run setup
```

To enter the environment interactively, run
```bash
pixi shell
```

### Format:
-----------
Coruscant provides Black-based formatting tasks.

To format the repository, run
```bash
pixi run format
```

To check formatting without modifying files, run
```bash
pixi run format-check
```

### Tests:
----------
`coruscant` runs its test suite with `pytest`.

To run the tests, use
```bash
pixi run test
```

