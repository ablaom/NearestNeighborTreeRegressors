"""
To list available datasets: 

    Datasets.list()

Loading example: 

   df = Datasets.load("ames_s")

"""
module Datasets

export load, list

import DataFrames: DataFrames, readtable

# to be extended:
import Base.show
import Base.Markdown.@md_str

immutable Dataset
    source::String
    startrow::Int
    nrows::Int
    features
    doc::Markdown.MD
end

dic = Dict{String,Dataset}()

### DATASET SPECIFICATIONS NOW FOLLOW. IF ALL FEATURES ARE TO BE USED, USE EMPTY `Symbol[]`

dic["ames_40"] = Dataset("/Users/anthony/Dropbox/AmesHousePrices/2.cleaned/train.csv",
                        1,
                        1456,
                         Symbol[:target,:OverallQual,:GrLivArea,:Neighborhood,:x1stFlrSF,:TotalBsmtSF,
                                :BsmtFinSF1,:LotArea,:GarageCars,:MSSubClass,:GarageArea,:YearRemodAdd,
                                :YearBuilt,:LotFrontage,:Exterior2nd,:x2ndFlrSF,:BsmtUnfSF,:OpenPorchSF,
                                :OverallCond,:HouseStyle,:MoSold,:MSZoning,:WoodDeckSF,:MasVnrArea,
                                :Exterior1st,:BsmtFinType1,:BsmtFinType2,:BedroomAbvGr,:EnclosedPorch,
                                :BsmtFullBath,:YrSold,:BsmtCond,:Fireplaces,:ScreenPorch,:GarageQual,
                                :LotConfig,:MasVnrType,:LandSlope,:Foundation,:SaleType,:Condition1],
                        md"*Ames House Price* data. Cleaned, and with the 40 most important features. Note that
                        `:target` is the natural logarithm of the original `:SalePrice` field.")

dic["ames_40r"] = Dataset("/Users/anthony/Dropbox/AmesHousePrices/2.cleaned/train_randomized.csv",
                         1,
                         1456,
                         Symbol[:target,:OverallQual,:GrLivArea,:Neighborhood,:x1stFlrSF,:TotalBsmtSF,
                                :BsmtFinSF1,:LotArea,:GarageCars,:MSSubClass,:GarageArea,:YearRemodAdd,
                                :YearBuilt,:LotFrontage,:Exterior2nd,:x2ndFlrSF,:BsmtUnfSF,:OpenPorchSF,
                                :OverallCond,:HouseStyle,:MoSold,:MSZoning,:WoodDeckSF,:MasVnrArea,
                                :Exterior1st,:BsmtFinType1,:BsmtFinType2,:BedroomAbvGr,:EnclosedPorch,
                                :BsmtFullBath,:YrSold,:BsmtCond,:Fireplaces,:ScreenPorch,:GarageQual,
                                :LotConfig,:MasVnrType,:LandSlope,:Foundation,:SaleType,:Condition1],
                        md"*Ames House Price* data. Cleaned, randomized, and with the 40 most important 
                           features + target. Note that `:target` is the natural logarithm of the 
                           original `:SalePrice` field.")


dic["ames_12r"] = Dataset("/Users/anthony/Dropbox/AmesHousePrices/2.cleaned/train_randomized.csv",
                         1,
                         1456,
                         Symbol[:target,:OverallQual,:GrLivArea,:Neighborhood,:x1stFlrSF,:TotalBsmtSF,
                                :BsmtFinSF1,:LotArea,:GarageCars,:MSSubClass,:GarageArea,:YearRemodAdd,
                                :YearBuilt],
                        md"*Ames House Price* data. Cleaned, randomized, and with the 12 most important 
                           features + target. Note that `:target` is the natural logarithm of the 
                           original `:SalePrice` field.")

dic["toyota_r"] = Dataset("/Users/anthony/Dropbox/Ensemble Regressors/0ToyotaCorolla/ToyotaCorollaRandomized.csv",
                          1,
                          1436,
                          [:Price,:Age,:KM,:FuelType,:HP,:MetColor,:Automatic,:CC,:Doors,:Weight],
                          md"*Toyota Corolla Price* data, randomized.")
                          
dic["bike_r"] = Dataset("/Users/anthony/Dropbox/Datasets/Bike-Sharing/2./hour.csv",
                        1,
                        17379,
                        Symbol[:cnt_log, :day_cos, :day_sin, :hr, :holiday,
                               :weekday, :workingday, :weathersit, :temp, :atemp, :hum,
                               :windspeed],
                        md"*Bike Sharing* dataset, randomized. The target is `:cnt_log`, the logarithm
                           of the number of bike rentals during an hour `:hr` on a day of the year specified
                           by `:day_cos` and `:day_sin`.")

dic["airfoil_r"] = Dataset("/Users/anthony/Dropbox/Datasets/Airfoil-Self-Noise/2./airfoi_self_noise.csv",
                           1,
                           1503,
                           [:frequency, :angle, :chord, :velocity, :displacement, :decibels],
                           md"*Airfoil Self-Noise* data, randomized. The target is the scaled sound pressure
                              in decibels, labeled `:decibels`. All attributes are ordinal.")

dic["forestfires_r"] = Dataset("/Users/anthony/Dropbox/Datasets/Forest-Fires/1./forestfires.csv",
                               1,
                               517,
                               Symbol[:area,:X,:Y,:month,:day,:FFMC,:DMC,:DC,:ISI,:temp,:RH,:wind,:rain],
                               md"*Forest Fires* dataset, randomized. The target is area of the burn, `:area`. 
                                 There are 12 input features, two of them categorical.")

dic["news_r"] = Dataset("/Users/anthony/Dropbox/Datasets/Online-News-Popularity/2./OnlineNewsPopularity.csv",
                        1,
                        39644,
                        Symbol[:shares,:n_tokens_title,:n_tokens_content,:n_unique_tokens,:n_non_stop_words,:n_non_stop_unique_tokens,:num_hrefs,:num_self_hrefs,:num_imgs,:num_videos,:average_token_length,:num_keywords,:data_channel_is_lifestyle,:data_channel_is_entertainment,:data_channel_is_bus,:data_channel_is_socmed,:data_channel_is_tech,:data_channel_is_world,:kw_min_min,:kw_max_min,:kw_avg_min,:kw_min_max,:kw_max_max,:kw_avg_max,:kw_min_avg,:kw_max_avg,:kw_avg_avg,:self_reference_min_shares,:self_reference_max_shares,:self_reference_avg_sharess,:weekday_is_monday,:weekday_is_tuesday,:weekday_is_wednesday,:weekday_is_thursday,:weekday_is_friday,:weekday_is_saturday,:weekday_is_sunday,:is_weekend,:LDA_00,:LDA_01,:LDA_02,:LDA_03,:LDA_04,:global_subjectivity,:global_sentiment_polarity,:global_rate_positive_words,:global_rate_negative_words,:rate_positive_words,:rate_negative_words,:avg_positive_polarity,:min_positive_polarity,:max_positive_polarity,:avg_negative_polarity,:min_negative_polarity,:max_negative_polarity,:title_subjectivity,:title_sentiment_polarity,:abs_title_subjectivity,:abs_title_sentiment_polarity],
                        md"*Online News Popularity* dataset, randomized. The target is the number of \"shares\" of the news article. There are 58 input features: 13 boolean, 45 ordinal. Citation request: K. Fernandes, P. Vinagre and P. Cortez. A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News. Proceedings of the 17th EPIA 2015 - Portuguese Conference on Artificial Intelligence, September, Coimbra, Portugal.")

dic["news_rs"] = Dataset("/Users/anthony/Dropbox/Datasets/Online-News-Popularity/2./OnlineNewsPopularity.csv",
                          1,
                          39644,
                          [:shares,:kw_avg_avg,:self_reference_avg_sharess,:kw_max_avg,:kw_avg_max,
                           :self_reference_min_shares,:n_tokens_content,:LDA_01,
                           :average_token_length,:global_subjectivity,:n_unique_tokens,
                           :LDA_04,:n_non_stop_unique_tokens],
                         md"*Online News Popularity* dataset, randomized and with a reduced set of features. The target is the number of \"shares\" of the news article. There are 12 input features (selected from the original 58), all ordinal. The most important features were selected using a tree-based ranking without regularization. Citation request: K. Fernandes, P. Vinagre and P. Cortez. A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News. Proceedings of the 17th EPIA 2015 - Portuguese Conference on Artificial Intelligence, September, Coimbra, Portugal.")

dic["abalone_r"] = Dataset("/Users/anthony/Dropbox/Datasets/Abalone/2./abalone.csv",
                           1,
                           4177,
                           [:rings, :sex, :length, :diameter, :height, :whole_wt, :shucked_wt, :viscera_wt,
                            :shell_wt],
                           md"*Abalone* dataset, randomized. The target is `:rings`. There are 8 input attributes,
                              1 of which is categorical.")

dic["power_r"] = Dataset("/Users/anthony/Dropbox/Datasets/Power-Plant/1./data.csv",
                         1,
                         9568,
                         [:PE,:AT,:V,:AP,:RH],
                         md"*Combined Cycle Power Plant* dataset, randomized. The target is the electrical power output of the plant, `:PE`. There are 4 input attributes, all ordinal. Citation requests: Pınar Tüfekci, Prediction of full load electrical power output of a base load operated combined cycle power plant using machine learning methods, International Journal of Electrical Power & Energy Systems, Volume 60, September 2014, Pages 126-140, ISSN 0142-0615, http://dx.doi.org/10.1016/j.ijepes.2014.02.027.
(http://www.sciencedirect.com/science/article/pii/S0142061514000908)

Heysem Kaya, Pınar Tüfekci , Sadık Fikret Gürgen: Local and Global Learning Methods for Predicting Power of a Combined Gas & Steam Turbine, Proceedings of the International Conference on Emerging Trends in Computer and Electronics Engineering ICETCEE 2012, pp. 13-18 (Mar. 2012, Dubai)")

dic["concrete_r"] = Dataset("/Users/anthony/Dropbox/Datasets/Concrete-Compressive-Strength/1./Concrete_Data.csv",
                            1,
                            1030,
                            [:strength,:cement,:slag,:ash,:water,:plasticizer,:coarse_aggregate,:fine_aggregate,:age],
                            md"*Concrete Compressive Strength* dataset, randomized. Target is concrete strength, `:strength`. There are 8 input features describing the composition of the concrete and its age, all ordinal. Cite: I-Cheng Yeh, Modeling of strength of high performance concrete using artificial neural networks, Cement and Concrete Research, Vol. 28, No. 12, pp. 1797-1808 (1998). ")
 
function show(stream::IO, d::Dataset)
    display(d.doc)
    println(stream, "source = $(d.source)")
end

function list()
    ordered_keys = sort(collect(keys(dic)))
    for k in ordered_keys
        println()
        println("\"$k\":")
        println()
        show(dic[k])
    end
end
                             
function load(handle::String)
    d = dic[handle]
    df = readtable(d.source)
    if d.features != Symbol[]
        return df[d.startrow:(d.startrow + d.nrows - 1), d.features]
    else
        return df[d.startrow:(d.startrow + d.nrows - 1),:]
    end
    
end

end # of module
