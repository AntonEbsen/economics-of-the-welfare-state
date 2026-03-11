# Data and Methods

## Data Sources

This study utilizes panel data from multiple sources:


**CPDS:**
- *Social security transfers* (`sstran`)
- *Government deficit* (`deficit`)
- *Government debt* (`debt`)

**World Bank:**
- *Log population* (`ln_population`)
- *Inflation rate* (`inflation_cpi`)
- *Age dependency ratio* (`dependency_ratio`)

**World Bank / OECD:**
- *Log GDP per capita* (`ln_gdppc`)

## Sample

The analysis covers **32 countries** over the period **1980–2023**, yielding **1,408 country-year observations**.


**Countries included:**
AUS AUT BEL BGR CAN CHE CZE DEU
DNK ESP EST FIN FRA GBR GRC HUN
IRL ISL ITA JPN LTU LUX LVA NLD
NOR NZL POL PRT SVK SVN SWE USA

## Panel Structure

The panel is **balanced**, with all countries observed in all years.

## Variable Construction


**Dependent and Independent Variables:**

- **Social security transfers** (sstran): Public social security transfers as percentage of GDP
- **Government deficit** (deficit): General government deficit as percentage of GDP. Negative values indicate surplus.
- **Government debt** (debt): Gross general government debt as percentage of GDP
- **Log population** (ln_population): Natural logarithm of total population
- **Log GDP per capita** (ln_gdppc): Natural logarithm of GDP per capita in constant 2015 USD
- **Inflation rate** (inflation_cpi): Annual percentage change in consumer price index
- **Age dependency ratio** (dependency_ratio): Ratio of dependents (people younger than 15 or older than 64) to working-age population (ages 15-64)

## Missing Data

Missing data patterns:

- sstran: 142 observations (10.1%)
- deficit: 157 observations (11.2%)
- debt: 180 observations (12.8%)
- ln_gdppc: 116 observations (8.2%)
- inflation_cpi: 97 observations (6.9%)
- dependency_ratio: 320 observations (22.7%)
- KOFGI: 68 observations (4.8%)
- KOFEcGI: 68 observations (4.8%)
- KOFSoGI: 68 observations (4.8%)
- KOFPoGI: 68 observations (4.8%)

## Summary Statistics


Table 1 presents summary statistics for all variables in the analysis.

*(Summary statistics table would be inserted here using `generate_summary_stats()`)*


## Estimation Strategy

The empirical analysis employs panel data methods with fixed effects to control for unobserved country-specific and time-specific heterogeneity. Standard errors are clustered at the country level to account for within-country correlation over time.


---
*Generated: 2026-03-11 13:35*
