# scythe
Automated questionnaire abbreviation in Python, as introduced and described in [Yarkoni (2010)](http://pilab.psy.utexas.edu/publications/Yarkoni_JRP_2010a.pdf). Scythe uses customizable genetic algorithms to rapidly abbreviate long questionnaire measures--often reducing their length by as much as 80 - 90% with relatively little loss of fidelity.

## Installation
Assuming Python and pip are installed, scythe can be installed from PyPI via the command line:
```
pip install scythe
```
Alternatively, for the latest (development) version, install directly from github:
```
pip install git+https://github.com/tyarkoni/scythe.git
```
#### Dependencies
Aside from standard scientific python packages (numpy, matplotlib, and pandas--all conveniently included in the [Anaconda](http://???) bundle), the only current dependency is [deap](https://github.com/DEAP/deap/), which can be installed from PyPI ("pip install deap").

## Quickstart
This example reproduces the core results in [Eisenbarth, Lilienfeld, & Yarkoni (2014)](http://pilab.psy.utexas.edu/publications/Eisenbarth_Psychological_Assessment_2014.pdf). For a more comprehensive and detailed walk-through, including generation of all the figures in the manuscript, see the [demo IPython notebook](https://github.com/tyarkoni/scythe/master/examples/PPI-R/PPI-R%20abbreviation.ipynb), which can be [rendered online](https://github.com/tyarkoni/scythe/blob/master/examples/PPI-R/PPI-R%20abbreviation.ipynb). All data needed to run the example below can be found in [examples/PPI-R/data](https://github.com/tyarkoni/scythe/tree/master/examples/PPI-R/data).

```python
import scythe

# Initialize the measure/questionnaire we want to abbreviate.
# We drop all rows with a missing value for at least one item.
ppi = scythe.Measure(X='data/PPI-R_German_data.txt', missing='drop')

# Generate scale scores using the PPI-R scoring key, providing names for the columns.
ppi.score(key='data/PPI-R_scoring_key.txt', columns=['B','Ca','Co','F','M','R','So','St'], rescale=True)

# Initialize a new measure generator
gen = scythe.Generator()

# Run the generator for 1000 generations
# We'll seed the random number generator to ensure deterministic results.
gen.run(ppi, n_gens=1000, seed=64)

# Save the resulting abbreviated version
abb_ppi = gen.abbreviate()
abb_ppi.save(prefix='abbreviated')
```
That's it! We should now have two text files in our working directory--one that provides a basic summary of the abbreviated measure, and one that contains a scoring key we can use to automatically score the abbreviated measure's scales using item scores for the original measure.

Scythe provides many more options, including the ability to customize many aspects of the evaluation and abbreviation process (e.g., to adjust the amount of desired abbreviation), as well as various plotting functions that can help us evaluate the quality of the result and track the evolutionary process over successive generations. For a more detailed walk-through of some of these features, see the [demo IPython notebook](https://github.com/tyarkoni/scythe/tree/master/examples/PPI-R/data).