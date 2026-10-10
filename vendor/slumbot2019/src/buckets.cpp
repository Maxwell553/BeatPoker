#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "board_tree.h"
#include "buckets.h"
#include "card_abstraction.h"
#include "files.h"
#include "game.h"
#include "io.h"

Buckets::Buckets(const CardAbstraction &ca, bool numb_only) {
  BoardTree::Create();
  int max_street = Game::MaxStreet();
  mapped_.resize(max_street + 1, nullptr);
  mapped_sizes_.resize(max_street + 1, 0);
  none_.reset(new bool[max_street + 1]);
  short_buckets_ = new unsigned short *[max_street + 1];
  int_buckets_ = new int *[max_street + 1];
  for (int st = 0; st <= max_street; ++st) {
    short_buckets_[st] = nullptr;
    int_buckets_[st] = nullptr;
  }
  char buf[500];
  num_buckets_.reset(new int[max_street + 1]);
  for (int st = 0; st <= max_street; ++st) {
    if (ca.Bucketing(st) == "none") {
      none_[st] = true;
      num_buckets_[st] = 0;
      continue;
    }
    none_[st] = false;
    sprintf(buf, "%s/num_buckets.%s.%i.%i.%i.%s.%i", Files::StaticBase(), Game::GameName().c_str(),
	    Game::NumRanks(), Game::NumSuits(), max_street, ca.Bucketing(st).c_str(), st);
    Reader reader(buf);
    num_buckets_[st] = reader.ReadIntOrDie();
  }

  if (! numb_only) {
    for (int st = 0; st <= max_street; ++st) {
      if (none_[st]) continue;
      int num_boards = BoardTree::NumBoards(st);
      int num_hole_card_pairs = Game::NumHoleCardPairs(st);
      // Need to use an unsigned int for this
      unsigned int num_hands = (unsigned int)num_boards * (unsigned int)num_hole_card_pairs;
      long long int lli_num_hands = num_hands;
  
      sprintf(buf, "%s/buckets.%s.%i.%i.%i.%s.%i", Files::StaticBase(), Game::GameName().c_str(),
	      Game::NumRanks(), Game::NumSuits(), max_street, ca.Bucketing(st).c_str(), st);
      Reader reader(buf);
      long long int file_size = reader.FileSize();
      const char *map_mode = getenv("SLUMBOT_MMAP_BUCKETS");
      if (map_mode && !strcmp(map_mode, "1") &&
          (file_size == lli_num_hands * 2 || file_size == lli_num_hands * 4)) {
        int fd = open(buf, O_RDONLY);
        if (fd < 0) { perror("open bucket map"); exit(-1); }
        void *address = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (address == MAP_FAILED) { perror("mmap bucket map"); exit(-1); }
        mapped_[st] = address;
        mapped_sizes_[st] = file_size;
        if (file_size == lli_num_hands * 2) short_buckets_[st] = (unsigned short *)address;
        else int_buckets_[st] = (int *)address;
        continue;
      }
      if (file_size == lli_num_hands * 2) {
	short_buckets_[st] = new unsigned short[num_hands];
	for (unsigned int h = 0; h < num_hands; ++h) {
	  short_buckets_[st][h] = reader.ReadUnsignedShortOrDie();
	}
      } else if (file_size == lli_num_hands * 4) {
	int_buckets_[st] = new int[num_hands];
	for (unsigned int h = 0; h < num_hands; ++h) {
	  int_buckets_[st][h] = reader.ReadIntOrDie();
	}
      } else {
	fprintf(stderr, "BucketsInstance::Initialize: Unexpected file size %lli\n", file_size);
	exit(-1);
      }
    }
  }
}

// Allow a dummy empty buckets object to be created
Buckets::Buckets(void) {
  short_buckets_ = nullptr;
  int_buckets_ = nullptr;
}

Buckets::~Buckets(void) {
  int max_street = Game::MaxStreet();
  if (short_buckets_) {
    for (int st = 0; st <= max_street; ++st) {
      if (short_buckets_[st] && (mapped_.empty() || !mapped_[st])) delete [] short_buckets_[st];
    }
    delete [] short_buckets_;
  }
  if (int_buckets_) {
    for (int st = 0; st <= max_street; ++st) {
      if (int_buckets_[st] && (mapped_.empty() || !mapped_[st])) delete [] int_buckets_[st];
    }
    delete [] int_buckets_;
  }
  for (size_t st = 0; st < mapped_.size(); ++st) {
    if (mapped_[st]) munmap(mapped_[st], mapped_sizes_[st]);
  }
}
