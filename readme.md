# Spectral Clustering with Population Fairness Constraints

### Overview
**Spectral Clustering** has gained popularity as an algorithm for graph
partitioning and clustering in recent years. It often performs better than
traditional clustering algorithms such as K-means clustering while being easy
to implement. One limitation of this method is that it may overlook the
distributions of underlying groups in the dataset. For example,
we could end up with a disproportionate number of males and females in the
clusters that we obtained when checking for communities in the residents of a city.

This could hinder or even completely prevent us from drawing useful inferences
from the data. An example of this would be when studying drug users in an area
coming from different ethnic backgrounds. If the clusters obtained are just
reflections of ethnicity, we have not learned anything new. This is where we
introduce **Fair Spectral Clustering**. Simply by adding a linear constraint,
we can ensure sufficient representation from each underlying group in
every cluster returned by the algorithm.

#### Resources
* General introduction to spectral clustering: [Tutorial on Spectral Clustering.](https://arxiv.org/abs/0711.0189)

* [Spectral Clustering with fairness constraints.](https://arxiv.org/abs/1901.08668)

#### Datasets
* Function for generating synthetic dataset can be found in source/utils/utils.py
* [FriendshipNet](http://www.sociopatterns.org/datasets/high-school-contact-and-friendship-networks/)
* [DrugNet](https://sites.google.com/site/ucinetsoftware/datasets/covert-networks/drugnet)
* [Facebook dataset](https://snap.stanford.edu/data/ego-Facebook.html)



### About
This repository was created as part of E0-259 (Data Analytics) course project by team 22.
#### Contributors
* Chinmay Ratnaparkhi
* Dhanus M Lal
* Suryansh Shrivastava
* Vikas Verma
