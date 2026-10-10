// Read-only access to a complete experimental ECFR checkpoint.
// Stdin: street hole0 hole1 [board cards for this street] root/action/...
// Cards use rank*4+suit. Stdout is one JSON object per request.
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "betting_abstraction.h"
#include "betting_abstraction_params.h"
#include "betting_tree.h"
#include "betting_trees.h"
#include "board_tree.h"
#include "canonical.h"
#include "cards.h"
#include "card_abstraction.h"
#include "card_abstraction_params.h"
#include "files.h"
#include "game.h"
#include "game_params.h"
#include "hand_tree.h"
#include "params.h"

struct Mapping {
  int fd=-1;
  size_t size=0;
  const void *data=nullptr;
  explicit Mapping(const std::string &path) {
    fd=open(path.c_str(),O_RDONLY);
    if(fd<0) throw std::runtime_error("Cannot open "+path);
    struct stat info;
    if(fstat(fd,&info)) throw std::runtime_error("Cannot stat mapping");
    size=info.st_size;
    if(size) {
      data=mmap(nullptr,size,PROT_READ,MAP_PRIVATE,fd,0);
      if(data==MAP_FAILED) throw std::runtime_error("Cannot map checkpoint");
    }
  }
  ~Mapping(){if(data && data!=MAP_FAILED) munmap(const_cast<void*>(data),size);if(fd>=0)close(fd);}
};

static uint32_t BoardCode(const Card *board,int count) {
  uint32_t code=0;
  for(int i=0;i<count;++i) code=code*52+board[i];
  return code;
}

class Policy {
 public:
  std::unique_ptr<BettingTrees> trees;
  std::array<unsigned,4> buckets={169,512,256,256};
  std::array<std::unique_ptr<Mapping>,4> cards;
  std::array<std::unique_ptr<Mapping>,8> strategies;
  std::array<size_t,8> lengths={};
  std::unordered_map<Node*,size_t> offsets;
  std::array<std::vector<std::pair<uint32_t,uint32_t>>,4> boards;

  Policy(const BettingAbstraction &ba,const std::string &checkpoint,int batch,const CardAbstraction *ca) {
    BoardTree::Create();
    trees.reset(new BettingTrees(ba));
    const char *names[4]={"null","potential512","hs256","hs256"};
    for(int st=0;st<4;++st) {
      std::string base="holdem.13.4.3."+(ca?ca->Bucketing(st):std::string(names[st]))+"."+std::to_string(st);
      std::ifstream input("num_buckets."+base,std::ios::binary);
      unsigned count=0;input.read(reinterpret_cast<char*>(&count),4);
      if(!input || count<1 || count>65536 || (st==0 && count!=169))
        throw std::runtime_error("Incompatible card abstraction");
      buckets[st]=count;
      cards[st].reset(new Mapping("buckets."+base));
      size_t expected=size_t(BoardTree::NumBoards(st))*Game::NumHoleCardPairs(st)*2;
      if(cards[st]->size!=expected)throw std::runtime_error("Incomplete card map");
      for(int b=0;b<BoardTree::NumBoards(st);++b)
        boards[st].push_back({BoardCode(BoardTree::Board(st,b),Game::NumBoardCards(st)),uint32_t(b)});
      std::sort(boards[st].begin(),boards[st].end());
      for(int p=0;p<2;++p) {
        std::string path=checkpoint+"/sumprobs.x.0.0."+std::to_string(st)+"."+std::to_string(batch)+".p"+std::to_string(p)+".i";
        strategies[p*4+st].reset(new Mapping(path));
      }
    }
    Index(trees->Root());
    for(int i=0;i<8;++i) if(lengths[i]*4!=strategies[i]->size)
      throw std::runtime_error("Checkpoint size does not match betting tree");
  }

  void Index(Node *node) {
    if(node->Terminal())return;
    int st=node->Street(),i=node->PlayerActing()*4+st,n=node->NumSuccs();
    if(!offsets.emplace(node,lengths[i]).second)throw std::runtime_error("Reentrant betting tree unsupported");
    if(n>1)lengths[i]+=size_t(buckets[st])*n;
    for(int a=0;a<n;++a)Index(node->IthSucc(a));
  }

  std::string Query(const std::string &line) {
    std::istringstream in(line);
    int st;Card raw_hole[2],raw_board[5]={},canon_hole[2],canon_board[5]={};
    if(!(in>>st>>raw_hole[0]>>raw_hole[1]) || st<0 || st>3)throw std::runtime_error("Invalid request");
    int count=Game::NumBoardCards(st);
    for(int i=0;i<count;++i)if(!(in>>raw_board[i]))throw std::runtime_error("Missing board");
    bool seen[52]={};
    for(int i=0;i<count+2;++i) {
      int c=i<2?raw_hole[i]:raw_board[i-2];
      if(c<0 || c>=52 || seen[c])throw std::runtime_error("Invalid cards");
      seen[c]=true;
    }
    std::string path,extra;
    if(!(in>>path) || (in>>extra) || path.substr(0,4)!="root")throw std::runtime_error("Invalid path");
    Node *node=trees->Root();
    std::istringstream actions(path.substr(4));
    std::string action;
    while(std::getline(actions,action,'/')) {
      if(action.empty())continue;
      int chosen=-1;
      for(int i=0;i<node->NumSuccs();++i)if(node->ActionName(i)==action){chosen=i;break;}
      if(chosen<0)throw std::runtime_error("Unknown action path");
      node=node->IthSucc(chosen);
    }
    std::ostringstream out;
    if(node->Terminal()) {
      out<<"{\"terminal\":true,\"showdown\":"<<(node->Showdown()?"true":"false")
         <<",\"payoff\":"<<node->LastBetTo()<<",\"winner\":"<<node->PlayerActing()<<"}";
      return out.str();
    }
    int node_st=node->Street();
    // Query ancestors with their own public cards, without future information.
    if(node_st>st)throw std::runtime_error("Decision needs more board cards");
    CanonicalizeCards(raw_board,raw_hole,node_st,canon_board,canon_hole);
    uint32_t code=BoardCode(canon_board,Game::NumBoardCards(node_st));
    auto &index=boards[node_st];
    auto it=std::lower_bound(index.begin(),index.end(),std::make_pair(code,uint32_t(0)));
    if(it==index.end() || it->first!=code)throw std::runtime_error("Canonical board not found");
    Card hand[7]={canon_hole[0],canon_hole[1]};
    for(int i=0;i<Game::NumBoardCards(node_st);++i)hand[i+2]=canon_board[i];
    size_t h=size_t(it->second)*Game::NumHoleCardPairs(node_st)+HCPIndex(node_st,hand);
    unsigned bucket=static_cast<const uint16_t*>(cards[node_st]->data)[h];
    if(bucket>=buckets[node_st])throw std::runtime_error("Invalid bucket");
    int p=node->PlayerActing(),n=node->NumSuccs();
    size_t offset=offsets.at(node)+size_t(bucket)*n;
    const int32_t *weights=static_cast<const int32_t*>(strategies[p*4+node_st]->data);
    out<<"{\"street\":"<<node_st<<",\"player\":"<<p<<",\"bet_to\":"<<node->LastBetTo()<<",\"bucket\":"<<bucket<<",\"actions\":[";
    for(int i=0;i<n;++i) {
      if(i)out<<",";
      const char *kind=i==node->CallSuccIndex()?"call":i==node->FoldSuccIndex()?"fold":"raise";
      out<<"{\"name\":\""<<node->ActionName(i)<<"\",\"kind\":\""<<kind<<"\",\"total_target\":"<<node->IthSucc(i)->LastBetTo()<<"}";
    }
    out<<"],\"counts\":[";
    for(int i=0;i<n;++i){if(i)out<<",";int32_t w=n==1?1:weights[offset+i];if(w<0)throw std::runtime_error("Negative count");out<<w;}
    out<<"]}";
    return out.str();
  }

  std::string Range(const std::string &line) {
    // Enumerate every possible private hand using public cards only. Ancestor
    // queries still use only that node's street when assigning card buckets.
    std::istringstream input(line);
    std::string command,path,extra;int st;
    if(!(input>>command>>st) || command!="range" || st<0 || st>3)
      throw std::runtime_error("Invalid range query");
    bool dead[52]={};std::ostringstream suffix;
    for(int i=0;i<Game::NumBoardCards(st);++i) {
      int c;if(!(input>>c) || c<0 || c>=52 || dead[c])throw std::runtime_error("Invalid public cards");
      dead[c]=true;suffix<<" "<<c;
    }
    if(!(input>>path) || (input>>extra))throw std::runtime_error("Invalid range path");
    suffix<<" "<<path;
    std::ostringstream output;output<<"{\"rows\":[";bool first=true;
    for(int a=0;a<52;++a)for(int b=a+1;b<52;++b) {
      if(dead[a] || dead[b])continue;
      if(!first)output<<",";first=false;
      output<<"{\"hole\":["<<a<<","<<b<<"],\"node\":"
            <<Query(std::to_string(st)+" "+std::to_string(a)+" "+std::to_string(b)+suffix.str())<<"}";
    }
    output<<"]}";return output.str();
  }
};

int main(int argc,char **argv) {
  try {
    if(argc!=5 && argc!=6)throw std::runtime_error("Usage: policy_server GAME BETTING CHECKPOINT_DIR BATCH [CARD_PARAMS]");
    Files::Init();auto gp=CreateGameParams();gp->ReadFromFile(argv[1]);Game::Initialize(*gp);
    auto bp=CreateBettingAbstractionParams();bp->ReadFromFile(argv[2]);BettingAbstraction ba(*bp);
    std::unique_ptr<CardAbstraction> ca;
    if(argc==6) {
      auto cp=CreateCardAbstractionParams();cp->ReadFromFile(argv[5]);
      ca.reset(new CardAbstraction(*cp));
    }
    Policy policy(ba,argv[3],std::stoi(argv[4]),ca.get());
    std::cout<<"{\"ready\":true,\"stack_bb\":"<<ba.StackSize()/2.0
             <<",\"betting_abstraction\":\""<<ba.BettingAbstractionName()<<"\""
             <<",\"card_abstraction\":\""<<(ca?ca->CardAbstractionName():"potential512_hs256")<<"\"}"<<std::endl;
    std::string line;
    while(std::getline(std::cin,line)) {
      try{std::cout<<(line.rfind("range ",0)==0?policy.Range(line):policy.Query(line))<<std::endl;}
      catch(const std::exception &e){std::cerr<<e.what()<<std::endl;std::cout<<"{\"error\":\"invalid policy query\"}"<<std::endl;}
    }
  } catch(const std::exception &e){std::cerr<<e.what()<<std::endl;return 2;}
}
