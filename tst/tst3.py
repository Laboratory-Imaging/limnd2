import limnd2, pandas

TEST_FILE = r"C:\Images\HtcTest\06_Translocation_v02.nd2"

with limnd2.Nd2Reader(TEST_FILE) as nd2:
    #print(nd2.results)

    df = nd2.result_private_table("06_Translocation_v01_20231115_b1932a", "side", "a").df

    print(df.head())

    df = df.query('(`Dose Response of NucCyto_Ratio (Avg)`.notna() & (0 < `Concentration`) & `NucCyto_Ratio (Avg)`.notna())')

    print(df)