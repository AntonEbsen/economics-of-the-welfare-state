# ============================================================================
# Panel Data Analysis in R
# Auto-generated script
# ============================================================================

# Install required packages (run once)
# install.packages(c('fixest', 'plm', 'did'))

# Load packages
library(fixest)
library(plm)
library(did)

# Load data
master <- read.csv('master.csv')
head(master)

# Variable descriptions:
# iso3: ISO 3166-1 alpha-3 country code - Three-letter country code
# year: Year - Calendar year of observation
# sstran: Social security transfers - Public social security transfers as percentage of GDP
# deficit: Government deficit - General government deficit as percentage of GDP. Negative values indicate surplus.
# debt: Government debt - Gross general government debt as percentage of GDP
# ln_population: Log population - Natural logarithm of total population
# ln_gdppc: Log GDP per capita - Natural logarithm of GDP per capita in constant 2015 USD
# inflation_cpi: Inflation rate - Annual percentage change in consumer price index
# dependency_ratio: Age dependency ratio - Ratio of dependents (people younger than 15 or older than 64) to working-age population (ages 15-64)

# Convert to panel data format (if using plm)
master_panel <- pdata.frame(master, index = c('iso3', 'year'))

# ============================================================================
# Sample Model Specifications
# ============================================================================

# Example: sstran as dependent variable

# 1. Pooled OLS
ols_model <- lm(sstran ~ deficit + debt + ln_population, data = master)
summary(ols_model)

# 2. Two-way fixed effects (fixest)
fe_model <- feols(sstran ~ deficit + debt + ln_population | iso3 + year, 
                   data = master, cluster = ~iso3)
summary(fe_model)

# 3. Two-way fixed effects (plm)
plm_model <- plm(sstran ~ deficit + debt + ln_population, 
                 data = master_panel, 
                 model = 'within', effect = 'twoways')
summary(plm_model)

# 4. Random effects
re_model <- plm(sstran ~ deficit + debt + ln_population, 
                data = master_panel, 
                model = 'random')
summary(re_model)

# 5. Hausman test (FE vs RE)
phtest(fe_model, re_model)

# ============================================================================
# Diagnostics
# ============================================================================

# Panel characteristics
pdim(master_panel)  # Panel dimensions

# Summary statistics
summary(master)
