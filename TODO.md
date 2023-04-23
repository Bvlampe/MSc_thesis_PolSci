# MSc Thesis PolSci: Forecasting terrorism

### Todo
- [ ] Homogenise country names
  - [ ] Create concordance tables (WIP)
  - [ ] Apply CTs to each dataset
- [ ] Decide on new datasets
  - [x] Weapons movement
  - [ ] Ties to US
- [ ] Train models
- [ ] Evaluate models

### WIP

- [ ] Add new datasets
  - [ ] Weapons movement
  - [ ] Ties to US (?)
- [ ] Create concordance tables
  - [x] List non-corresponding country names per dataset
  - [ ] Attribute missing country names from other datasets to those in GTD in Excel (manually)
  - [ ] Create dict or DataFrame from newly "sorted" file
- [ ] Finish calc_rel_frag() function

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