
using DataFrames, Preprocess

df = readtable("/Users/anthony/Dropbox/Datasets/Bike-Sharing-Dataset/raw/hour.csv")

# create numeric day field:
day1 = Date(df[1,:dteday]).instant.period.value
df[:day] = map(df[:dteday]) do x
    Date(x).instant.period.value - day1 + 1
end
# replacement for seasonal field:
df[:day_sin] = map(df[:day]) do x
    sin(2pi*x/365.25)
end
df[:day_cos] = map(df[:day]) do x
    cos(2pi*x/365.25)
end

df = df[:instant, :cnt,	:day, :day_cos, :day_sin, :hr, :holiday, :weekday,
        :workingday, :weathersit, :temp, :atemp, :hum, :windspeed, :casual,
        :registered]
