// Export the ECFR preflop checkpoint in its exact depth-first storage order.
// Card IDs in this library are rank*4+suit, unlike the Python engine.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include "betting_abstraction.h"
#include "betting_abstraction_params.h"
#include "betting_tree.h"
#include "betting_trees.h"
#include "files.h"
#include "game.h"
#include "game_params.h"
#include "params.h"

static std::ifstream streams[2];
static bool first=true;
static bool inspect=false;
static unsigned long long decision_values[4]={};
static unsigned long long forced[4]={};

static void Count(Node *node) {
  if (node->Terminal()) return;
  if (node->NumSuccs()>1) decision_values[node->Street()] += node->NumSuccs();
  else ++forced[node->Street()];
  for (int i=0;i<node->NumSuccs();++i) Count(node->IthSucc(i));
}

static void Walk(Node *node,const std::string &path) {
  if (node->Terminal() || node->Street()>0) return;
  if (!first) std::printf(",");
  first=false;
  int n=node->NumSuccs(),p=node->PlayerActing();
  std::printf("\n\"%s\":{\"player\":%d,\"bet_to\":%d,\"actions\":[",path.c_str(),p,node->LastBetTo());
  for (int i=0;i<n;++i) {
    if (i) std::printf(",");
    const char *kind=i==node->CallSuccIndex()?"call":i==node->FoldSuccIndex()?"fold":"raise";
    std::printf("{\"name\":\"%s\",\"kind\":\"%s\",\"target\":%d}",
                node->ActionName(i).c_str(),kind,node->IthSucc(i)->LastBetTo());
  }
  std::printf("],\"counts\":[");
  if (!inspect) for (int b=0;b<169;++b) {
    if (b) std::printf(",");
    std::printf("[");
    for (int i=0;i<n;++i) {
      int value=0;
      if (n>1) {
        streams[p].read(reinterpret_cast<char*>(&value),sizeof(value));
        if (!streams[p] || value<0) { std::fprintf(stderr,"Invalid checkpoint at %s\n",path.c_str()); std::exit(2); }
      }
      if (i) std::printf(",");
      std::printf("%d",value);
    }
    std::printf("]");
  }
  std::printf("]}");
  for (int i=0;i<n;++i) Walk(node->IthSucc(i),path+"/"+node->ActionName(i));
}

int main(int argc,char **argv) {
  if (argc!=5) { std::fprintf(stderr,"Usage: export_preflop GAME BETTING CHECKPOINT_DIR|- BATCH\n");return 2; }
  Files::Init();
  auto gp=CreateGameParams();gp->ReadFromFile(argv[1]);Game::Initialize(*gp);
  auto bp=CreateBettingAbstractionParams();bp->ReadFromFile(argv[2]);BettingAbstraction ba(*bp);
  BettingTrees trees(ba);
  inspect=std::string(argv[3])=="-";
  if (!inspect) for (int p=0;p<2;++p) {
    std::string filename=std::string(argv[3])+"/sumprobs.x.0.0.0."+argv[4]+".p"+std::to_string(p)+".i";
    streams[p].open(filename,std::ios::binary);
    if (!streams[p]) {std::fprintf(stderr,"Cannot open %s\n",filename.c_str());return 2;}
  }
  Count(trees.Root());
  std::printf("{\"format\":1,\"stack_bb\":%.1f,\"big_blind\":2,\"batch\":%d,\"validated_pro_level\":false,\"decision_values_per_bucket\":[%llu,%llu,%llu,%llu],\"forced_nodes\":[%llu,%llu,%llu,%llu],\"nodes\":{",
      ba.StackSize()/2.0,std::atoi(argv[4]),decision_values[0],decision_values[1],decision_values[2],decision_values[3],forced[0],forced[1],forced[2],forced[3]);
  Walk(trees.Root(),"root");
  std::printf("}}\n");
  if (!inspect) for (auto &stream:streams) {
    if (stream.peek()!=std::char_traits<char>::eof()) {std::fprintf(stderr,"Trailing checkpoint values\n");return 2;}
  }
}
