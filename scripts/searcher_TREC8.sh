#!/bin/bash
# Generate the properties file and consequently execute the CollectionSearcher

cd ../

homepath=`eval echo ~$USER`

#stopFilePath="$homepath/smart-stopwords"
stopFilePath="/store/TCD/TREC_CS/RetrievalModel/WebDoc_Indexer_Retriever_QE-master/smart-stopwords"
TRECCSTagsClustersFilePath="/store/Data/TREC_CS/queryTags_rated_3_4_Uniq_113_Phase2_clusters"
#TRECCSTagsClustersWeightFilePath="/store/Data/TREC_CS/queryWiseTagsWeightPhase2"
TRECCSTagsClustersWeightFilePath="/store/Data/TREC_CS/queryWiseTagsWeightPhase1"
#W2VFilePath="/store/TCD/TREC_CS/Wor2vec/trunk/64K_BOW_analysed1_vectors.bin.vec"
#W2VFilePath="/store/TCD/TREC_CS/Wor2vec/trunk/TREC8_BOW_vectors.bin.vec"
W2VFilePath="/store/TCD/TREC_CS/Wor2vec/trunk/TRECAdhoc_BOW_vectors.bin.vec"
#W2VFilePath="/store/Data/TRECAdhoc/TRECAdhoc_lexicon_BERTVecSubset.txt"
#W2VFilePath="/store/Data/TRECAdhoc/contextsQuerywise_left_5_TREC8_BERTVecAvg.txt"
#BERTFilePath="/store/Data/TRECAdhoc/contextsQuerywise_left_5_TREC6_BERTVecAvg.txt"
BERTFilePath="/store/Data/TRECAdhoc/contextsQuerywise_left_10_TREC8_BERTVecAvg.txt"
#BERTFilePath="/store/Data/TRECAdhoc/contextsQuerywise_lr_10_TREC6_BERTVecAvg.txt"
#queryWiseUserPrefFilePath="/store/Data/TREC_CS/qID_prefereceIDs"
#queryWiseUserPrefFilePath="/store/Data/TREC_CS/qID_prefereceIDs_Phase2"

queryWiseUserPrefFilePath="/store/Data/TREC_CS/qID_prefereceIDs_ClusterNo_Rating_Phase1_all"
#queryWiseUserPrefFilePath="/store/Data/TREC_CS/qID_prefereceIDs_ClusterNo_Rating_Phase1_withBlanks_havingTags"
#queryWiseUserPrefFilePath="/store/Data/TREC_CS/qID_prefereceIDs_ClusterNo_Rating_Phase1_allRating_havingTags"
#queryWiseUserPrefFilePath="/store/Data/TREC_CS/qID_prefereceIDs_ClusterNo_Rating_Phase2_havingTags"

#queryWiseUserPrefFilePath="/store/Data/TREC_CS/qID_prefereceIDs_allRatings"
#queryWiseUserPrefFilePath="/store/Data/TREC_CS/qID_prefereceIDs_MissingTags"
#queryWiseUserPrefFilePath="/store/Data/TREC_CS/qID_prefereceIDs_HavingTags"
#queryWiseUserPrefTagsFilePath="/store/Data/TREC_CS/qID_prefereceTags"
queryWiseUserPrefTagsFilePath="/store/Data/TREC_CS/qID_prefereceTags_Phase2"

queryWiseUserPrefNegativeTagsFilePath="/store/Data/TREC_CS/topic_61_Phase1_NegativeTags"
#queryWiseUserPrefNegativeTagsFilePath="/store/Data/TREC_CS/topic_58_Phase2_NegativeTags"

#foursquareDataFilePath="/store/Data/TREC_CS/Extra/Foursquare/fromMostafa/mostafa228778_and_ani969_uniq.json"
foursquareDataFilePath="/store/Data/TREC_CS/Extra/Foursquare/fromMostafa/FoursquareData_candidatePOIS_And_UserPrefs_in63257.json"

treccsQueryJsonFilePath="/store/Data/TREC_CS/topic_61_Phase1.json"


if [ ! -f $stopFilePath ]
then
    echo "Please ensure that the path of the stopword-list-file is set in the .sh file."
else
    echo "Using stopFilePath="$stopFilePath
fi
resPath=$homepath

echo $#
if [ $# -le 4 ] 
then
    echo "Usage: " $0 " list-of-arguments";
    echo "1. indexPath: Path of the index";
    echo "2. fieldToSearch: Field name of the index to be used for searching (full-content / content)";
    echo "3. queryPath: path of the query file (in proper xml format)"
    echo "4. queryFieldFlag: 1-title, 2-title+desc, 3-title+desc+narr";
    echo "5. similarityFunction: 0.DefaultSimilarity, 1.BM25, 2.LMJelinekMercer, 3.LMDirichlet, 4. DFR, 5. Multi (BM25+LM-JM+LM-Di)";
    echo "6. [param1]: 'k1'-BM25; lambda-LMJM; mu-LMDi; BasicModel-DFR";
    echo -e "\tDFR(BasicModel) - 1-BE, 2-D, 3-G, 4-IF, 5-In, 6-Ine, 7-P";
    echo "7. [param2]: optional 'b' - BM25; AfterEffect - DFR";
    echo -e "\tDFR(AfterEffect) - 1-B, 2-L";
    echo "8. [param3]: optional 'b' - BM25; Norma. - DFR";
    echo -e "\tDFR(Norma.) - 1-H1, 2-H2, 3-H3, 4-Z, 5-NoNorm";
    exit 1;
fi

indexPath=$1
fieldToSearch=$2
queryPath=$3
qff=$4
sf=$5

prop_name="webdoc-searcher.properties"
#echo $prop_name

#cd build/classes

# making the .properties file
cat > $prop_name << EOL

stopFilePath=$stopFilePath

indexPath=$indexPath

queryPath=$queryPath

TRECCSTagsClustersFilePath=$TRECCSTagsClustersFilePath
TRECCSTagsClustersWeightFilePath=$TRECCSTagsClustersWeightFilePath
W2VFilePath=$W2VFilePath
BERTFilePath=$BERTFilePath
queryWiseUserPrefFilePath=$queryWiseUserPrefFilePath
queryWiseUserPrefTagsFilePath=$queryWiseUserPrefTagsFilePath
queryWiseUserPrefNegativeTagsFilePath=$queryWiseUserPrefNegativeTagsFilePath
foursquareDataFilePath=$foursquareDataFilePath
treccsQueryJsonFilePath=$treccsQueryJsonFilePath


## queryFields
# 1: title
# 2: title + desc
# 3. title + desc + narr
queryFieldFlag=$qff

## Field to search in the index
fieldToSearch=$fieldToSearch

resPath=$resPath

### Similarity functions:
#0 - DefaultSimilarity
#1 - BM25Similarity
#2 - LMJelinekMercerSimilarity
#3 - LMDirichletSimilarity
similarityFunction=$sf

numHits=1000
#numHits=50

EOL
if [ $# -ge 6 ]
then
    cat >> $prop_name << EOL
param1=$6
EOL
fi


if [ $# -eq 7 ]
then
    cat >> $prop_name << EOL
param2=$7
EOL
fi

if [ $# -eq 8 ]
then
    cat >> $prop_name << EOL
param3=$8
EOL
fi

if [ $# -eq 9 ] # Special case: Multi (parameters order: <k1> <b> (BM25); <lamda> (LM-JM); <mu> (LM-Di))
then
    cat >> $prop_name << EOL
param2=$7
param3=$8
param4=$9
EOL
fi

echo $PWD
java -Xmx1g -cp $CLASSPATH:dist/WebData.jar:./lib/* searcher.WebDocSearcher_TRECCS_Novel $prop_name

