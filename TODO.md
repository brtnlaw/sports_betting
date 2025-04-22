## TODOs
- [ ] Obtain basic historical market data with payouts
- [ ] Obtain market data for derivative lines  
- [ ] Obtain weather data
- [ ] Potential helper function to get the transformed df right before model
- [ ] Given contributions from each column, harken back to the grouping

## Ideas
- Improve feature selection process, such as via SHAP, forward elimination, backward elimination
    - This way, we can see if a model ever "picks up" a certain feature
    - For basics, see: SelectKBest prior to model
- "Explosiveness"

## Done
- [x] Variable betting size based on opportunity (stellage)
- [x] Allow argsparser to use betting function