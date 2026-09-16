# Raw NASA Exoplanet data

`exoplanets.csv` is a reproducible snapshot of selected columns from the NASA
Exoplanet Archive Planetary Systems Composite Parameters (`pscomppars`) table.
It was downloaded on 2026-09-16 with `code/datasets/download_data.py`.

The snapshot is committed so that training and the assignment demonstration do
not depend on network access. Run `make download` to deliberately refresh it.
The archive changes as new confirmed planets and measurements are published, so
a refreshed snapshot can produce different metrics.

Source documentation:

- https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html
- https://exoplanetarchive.ipac.caltech.edu/docs/API_resources.html

NASA Exoplanet Archive DOI: `10.26133/NEA2`.

