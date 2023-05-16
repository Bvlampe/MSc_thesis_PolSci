# MSc Thesis PolSci: Forecasting terrorism

### Todo

- [ ] Implement attribution of group interventions (NATO, UN, etc.)
- [ ] Properly shift GTD data in a way that respects country-level index differences


---
### WIP

- [ ] DataPrep
  - [x] Finish generic_table_transform() function definition
  - [x] Merge datasets
  - [ ] Add new datasets
    - [ ] Weapons movement
    - [ ] Ties to US (?)
  - [ ] Interpolate values:
    - [ ] Fragility?
    - [ ] Inequality
    - [ ] Poverty
    - [ ] Literacy
    - [ ] Internet users?
    - [ ] Education
  - [ ] Format intervention and FH datasets
- [ ] Train models
  - [x] Linear regression
  - [x] Random forest
  - [x] GBM
- [ ] Evaluate models
  - [ ] Compare predictive power of variables

---
### Done ✓
- [x] Make main dataset with [country, year] multi-index
- [x] Change election system dataset to country-year format
- [x] Get all main datasets and add to project
  - [x] GTD
  - [x] Fragility
  - [x] Durability
  - [x] Electoral system
  - [x] Level of democracy
  - [x] Political rights
  - [x] Civil liberties
  - [x] Inequality
  - [x] Poverty
  - [x] Inflation
  - [x] Literacy
  - [x] Internet usage
  - [x] Interventions
  - [x] Religious fragmentation
  - [x] Globalisation
- [x] Create concordance tables
  - [x] List non-corresponding country names per dataset
  - [x] Attribute missing country names from other datasets to those in GTD in Excel (manually)
  - [x] Create dict or DataFrame from newly "sorted" file
- [x] Finish calc_rel_frag() function
- [x] Homogenise country names
  - [x] Create concordance tables (WIP)
  - [x] Apply CTs to each dataset
- [x] Decide on new datasets
  - [x] Weapons movement
  - [x] Ties to US