* ===========================================================================
* Panel Data Analysis
* Auto-generated Stata script
* ===========================================================================

* Load data
use "master.dta", clear

* Variable labels
label variable iso3 "ISO 3166-1 alpha-3 country code"
label variable year "Year"
label variable sstran "Social security transfers"
label variable deficit "Government deficit"
label variable debt "Government debt"
label variable ln_population "Log population"
label variable ln_gdppc "Log GDP per capita"
label variable inflation_cpi "Inflation rate"
label variable dependency_ratio "Age dependency ratio"

* Declare panel structure
encode iso3, gen(country_id)
xtset country_id year
xtdescribe

* Sample regressions

* 1. Pooled OLS
reg sstran deficit debt ln_population

* 2. Fixed effects
xtreg sstran deficit debt ln_population, fe vce(cluster country_id)

* 3. Random effects
xtreg sstran deficit debt ln_population, re

* 4. Hausman test
hausman ., sigmamore
