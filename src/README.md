Rhy Yan review notes:
I think we need to review the splitting logic in data_module.py. 
I'm not sure why there's so many reassignments going on here on the same variables:
```
       # Extract feature columns by dropping label columns
        X = self.data.copy()
        X = self.data.drop(columns=label_col)

        # Extract label columns, labels are what we want to predict using the model.
        Y = self.data.copy()
        Y = self.data[label_col]
        Y = Y.rename(columns={label_col[0]: "label"})
```

also i'd like to improve the naming on the filenames, like everything is processing data so i have no idea what data_module means on first glance.