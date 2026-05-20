# coruscant 
---------------------------------------------------------

"There are people from a thousand worlds on Coruscant. Worlds that have faced similar struggles. You may see it as a city of metal, but there is wisdom to be found here. And allies." - *Qui-Gon Jinn*

### A toolkit package for astronomical and cosmological analysis

-----------------------------------------------------------------

Coruscant, the world city, the ecumenopolis, capital of the Old Republic. And also my personal place to store code I find useful for cosmological pipelines, analysis, papers, plots, and more. To be built as I go !


### Installation: 
------------------

Coruscant installs into its own dedicated Conda environment named `coruscant`.
The environment is defined in `environment.yaml` and installs the package in editable mode with the development tools.

To create or recreate that environment, run
```bash
make install
```

### Format:
-----------
coruscant provides a format checker. 
This format checker is checked on every pull / push request during unittests.

### Tests:
----------
`coruscant` provides unit tests with the `unittests` module. 

