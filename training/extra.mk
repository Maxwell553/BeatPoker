bin/export_preflop: ../../training/export_preflop.cpp $(OBJS) $(HEADS)
	$(CXX) $(CFLAGS) -Isrc -o $@ $< $(OBJS) $(LIBRARIES)

bin/policy_server: ../../training/policy_server.cpp $(OBJS) $(HEADS)
	$(CXX) $(CFLAGS) -Isrc -o $@ $< $(OBJS) $(LIBRARIES)

bin/river_features: ../../training/river_features.cpp ../../training/river_features.h $(OBJS) $(HEADS)
	$(CXX) $(CFLAGS) -Isrc -o $@ $< $(OBJS) $(LIBRARIES)

bin/turn_features: ../../training/turn_features.cpp ../../training/turn_features.h ../../training/river_features.h $(OBJS) $(HEADS)
	$(CXX) $(CFLAGS) -Isrc -o $@ $< $(OBJS) $(LIBRARIES)

bin/flop_features: ../../training/flop_features.cpp ../../training/flop_features.h ../../training/river_features.h $(OBJS) $(HEADS)
	$(CXX) $(CFLAGS) -Isrc -o $@ $< $(OBJS) $(LIBRARIES)

bin/cluster_river: ../../training/cluster_river.cpp $(OBJS) $(HEADS)
	$(CXX) $(CFLAGS) -Isrc -o $@ $< $(OBJS) $(LIBRARIES)

bin/build_river_map: ../../training/build_river_map.cpp ../../training/river_features.h $(OBJS) $(HEADS)
	$(CXX) $(CFLAGS) -Isrc -o $@ $< $(OBJS) $(LIBRARIES)

bin/build_turn_map: ../../training/build_river_map.cpp ../../training/turn_features.h ../../training/river_features.h $(OBJS) $(HEADS)
	$(CXX) $(CFLAGS) -DTURN_FEATURES -Isrc -o $@ $< $(OBJS) $(LIBRARIES)

obj/ecfr_reference.o: src/ecfr.cpp $(HEADS)
	$(CXX) $(CFLAGS) -DECFR_HEAP_SCRATCH -c -o $@ $<

bin/run_ecfr_reference: obj/run_ecfr.o obj/ecfr_reference.o $(filter-out obj/ecfr.o,$(OBJS)) $(HEADS)
	$(CXX) $(CFLAGS) -o $@ obj/run_ecfr.o obj/ecfr_reference.o $(filter-out obj/ecfr.o,$(OBJS)) $(LIBRARIES)

bin/run_ecfr_optimized: obj/run_ecfr.o $(OBJS) $(HEADS)
	$(CXX) $(CFLAGS) -o $@ obj/run_ecfr.o $(OBJS) $(LIBRARIES)
