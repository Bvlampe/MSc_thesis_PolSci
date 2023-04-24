# MSc Thesis PolSci: Forecasting terrorism

### Todo
- [ ] Interpolate values for literacy dataset
- [ ] Format intervention dataset
- [ ] Add new datasets
  - [ ] Weapons movement
  - [ ] Ties to US (?)
- [ ] Homogenise country names
  - [x] Create concordance tables (WIP)
  - [ ] Apply CTs to each dataset
- [ ] Decide on new datasets
  - [x] Weapons movement
  - [ ] Ties to US
- [ ] Train models
- [ ] Evaluate models

---
### WIP

- [ ] DataPrep
  - [x] Finish generic_table_transform() function definition
  - [ ] Merge datasets

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