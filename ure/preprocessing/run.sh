# Remember to change the corresponding paths in `config.yml`
TRAIN_PATH=data/nyt/train.txt
DEV_PATH=data/nyt/dev.txt
TEST_PATH=data/nyt/test.txt
VOCAB_FILE=data/nyt/dict.word
ENTITY=data/nyt/dict.entity
ENTITY_FREQUENCY=data/nyt/dict.ent_wf
ENTITY_TYPE=data/nyt/dict.enttype
RELATION_PATH=data/nyt/dict.relation
# Remove NA instances
awk -F '\t' '{if($9!="") print $0}' $DEV_PATH > $DEV_PATH.filtered
awk -F '\t' '{if($9!="") print $0}' $TEST_PATH > $TEST_PATH.filtered
# or "no_relation" for TACRED
# awk -F '\t' '{if($9!="no_relation") print $0}' $TEST_PATH > $TEST_PATH.filtered
# Build vocabulary
# Build Entity type vocab
cut -f4 $TRAIN_PATH | tr "-" "\n" | sort | uniq > $ENTITY_TYPE
# Build word vocab
cut -f7 -d$'\t' $TRAIN_PATH | tr " " "\n" | sort | uniq > $VOCAB_FILE
# Build entity vocab & frequency
awk -F '\t' '{printf("%s\n%s\n", $2, $3)}' $TRAIN_PATH | sort | uniq -c | sed 's/^[ ]\+\([0-9]\+\) /\1\t/g' | awk -F '\t' '{printf("%s\t%s\n", $2, $1)}' > $ENTITY_FREQUENCY
cut -f1 $ENTITY_FREQUENCY > $ENTITY
# Build relation vocab
cut -f9 $TRAIN_PATH | grep . | sort | uniq > $RELATION_PATH
