import pandas as pd


def collectDataForASymbol(startYear, endYear, symbol, scaler):
    accumulated_df = pd.DataFrame()
    for y in range(startYear, endYear+1):
        dataset = pd.read_csv("clean-data/DSE-"+str(y)+".csv",low_memory=False)
        filtered_df = dataset[dataset['Scrip'] == symbol]
        if filtered_df.empty:
            continue
        filtered_df = filtered_df.sort_values(by=['DayIndex'])
        accumulated_df =  pd.concat([accumulated_df, filtered_df], ignore_index=True)

    cols = ["Open","High","Low","Close","Volume","DayIndex"]
    
    accumulated_df[cols] = scaler.fit_transform(accumulated_df[cols])

    return accumulated_df