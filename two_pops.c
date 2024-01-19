initialize() {
    if (!exists("dry_run"))
        defineConstant("dry_run", F);
    if (!exists("verbosity"))
        defineConstant("verbosity", 2);

    // Scaling factor to speed up simulation.
    // See SLiM manual:
    // `5.5 Rescaling population sizes to improve simulation performance`.
    defineConstant("Q", 2);

    defineConstant("burn_in", 1.0);
    defineConstant("generation_time", 2);
	//bird generation time is ~2 
    defineConstant("pop_names", c("SP1", "SP2", "Anc"));

    _recombination_rates = c(
        1.4333142270041838e-08);
    if (Q != 1) {
        _recombination_rates = (1-(1-2*_recombination_rates)^Q)/2;
    }
    _recombination_ends = c(
        9999);
    defineConstant("recombination_rates", _recombination_rates);
    defineConstant("recombination_ends", _recombination_ends);
    // whatever is in this dictionary will be saved out in the .trees file
    defineConstant("metadata", Dictionary("Q", Q));

    initializeMutationType(0, 0.5, "f", Q * 0);
    initializeGenomicElementType(0, c(0), c(1.0));
    initializeGenomicElement(0, c(
        0, 5100), c(
        4949, 9999));
    initializeMutationType(1, 0.5, "f", Q * 0);
    initializeMutationType(2, 0.5, "g", Q * -100, 10);
    initializeGenomicElementType(1, 1, 1);
    initializeGenomicElementType(2, 2, 1);
    initializeGenomicElement(2, 4950, 5049);
    initializeMutationRate(Q * 1.652e-08);

    // Time of epoch boundaries, in years before present.
    // The first epoch spans from INF to _T[0].
    defineConstant("_T", c(30000000, 0));

    // Population sizes in each epoch.
    _N = array(c(
        // INF:_T[0], _T[0]:_T[1], etc.
        c(0, 50000), // SP1
        c(0, 50000), // SP2
		c(100000, 100000) // ANC
    ), c(2, 3));

    defineConstant("num_epochs", length(_T));
    defineConstant("num_populations", ncol(_N));

    // Population growth rates for each epoch.
    defineConstant("growth_rates", array(c(
        // INF:_T[0], _T[0]:_T[1], etc.
		c(0.0, 0.0), // SP1
        c(0.0, 0.0), // SP2
        c(0.0, 0.0) // ANC
    ), c(num_epochs, num_populations)));

    no_migration = rep(0, num_populations*num_populations);

    // Migration rates for each epoch.
    // Migrations involving a population with size=0 are ignored.
    // XXX: document what the rows & cols correspond to.
    defineConstant("migration_matrices", array(c(

        // INF:_T[0]
        no_migration,

        // _T[1]:_T[2]
        no_migration

    ), c(num_populations, num_populations, num_epochs)));

    // Population splits, one row for each event.
    defineConstant("subpopulation_splits", array(c(
        // time, newpop, size, oldpop
        c(_T[0], 0, _N[1,0], 2),
        c(_T[0], 1, _N[1,1], 2)
    ), c(4, 2)));

    // Admixture pulses, one row for each pulse.
    defineConstant("admixture_pulses", c());

    // Drawn mutations, one row for each mutation.
    defineConstant("drawn_mutations", c());

    // Fitness callbacks, one row for each callback.
    defineConstant("fitness_callbacks", c());

    defineConstant("op_types", c("<", "<=", ">", ">="));
    // Allele frequency conditioning, one row for each.
    defineConstant("condition_on_allele_frequency", c());

    // One row for each sampling episode.
    defineConstant("sampling_episodes", array(c(
        // pop, n_inds, time
        c(0, 500, 0),
        c(1, 500, 0)
    ), c(3, 2)));

    defineConstant("N", asInteger(_N/Q));

    initializeRecombinationRate(recombination_rates, recombination_ends);
}

function (void)err(string$ s) {
    stop("ERROR: " + s);
}

function (void)warn(string$ s) {
    catn("WARNING: " + s);
}

function (void)dbg(string$ s, [integer$ debug_level = 2]) {
    if (verbosity >= debug_level) {
        catn(community.tick + ": " + s);
    }
}



// Check that sizes aren't dangerously low or zero (e.g. due to scaling).
function (void)check_size(integer$ pop, integer$ size, integer$ t) {
    if (size == 0) {
        err("The population size of p"+pop+" ("+pop_names[pop]+") is zero " +
            "at tick "+t+".");
    } else if (size < 50) {
        warn("p"+pop+" ("+pop_names[pop]+") has only "+size+" individuals " +
             "alive at tick "+t+".");
    }
}

// Return the epoch index for generation g.
function (integer)epoch(integer G, integer $g) {
    for (i in 0:(num_epochs-1)) {
        if (g < G[i]) {
            return i;
        }
    }
    return num_epochs - 1;
}

// Return the population size of pop at generation g.
function (integer)pop_size_at(integer G, integer$ pop, integer$ g) {
    e = epoch(G, g);
    N0 = N[e,pop];
    r = Q * growth_rates[e,pop];
    if (r == 0) {
        N_g = N0;
    } else {
        g_diff = g - G[e-1];
        N_g = asInteger(round(N0*exp(r*g_diff)));
    }
    return N_g;
}

// Return the number of generations that separate t0 and t1.
function (integer)gdiff(numeric$ t0, numeric t1) {
    return asInteger(round((t0-t1)/generation_time/Q));
}

// Output tree sequence file and end the simulation.
function (void)end(void) {
    sim.treeSeqOutput(trees_file, metadata=metadata);
    sim.simulationFinished();
}


// Add `mut_type` mutation at `pos`, to a single individual in `pop`.
function (void)add_mut(object$ mut_type, object$ pop, integer$ pos) {
   targets = sample(pop.genomes, 1);
   targets.addNewDrawnMutation(mut_type, pos);
}

// Return the allele frequency of a drawn mutation in the specified population.
// Assumes there's only one mutation of the given type.
function (float$)af(object$ mut_type, object$ pop) {
    mut = sim.mutationsOfType(mut_type);
    if (length(mut) == 0) {
        return 0.0;
    }
    return sim.mutationFrequencies(pop, mut);
}

// Save the state of the simulation.
function (void)save(void) {
    if (sim.getValue("restore_function")) {
        // Don't save if we're in the restore() function.
        return;
    }
    n_saves = 1 + sim.getValue("n_saves");
    sim.setValue("n_saves", n_saves);
    dbg("save() "+n_saves);
    sim.treeSeqOutput(trees_file, metadata=metadata);
}

// Restore the simulation state.
function (void)restore(void) {
    g_restore = community.tick;
    n_restores = 1 + sim.getValue("n_restores");
    sim.setValue("n_restores", n_restores);
    n_saves = sim.getValue("n_saves");
    if (n_saves == 0) {
        err("restore() in tick "+g_restore+", but nothing is saved.");
    }
    sim.readFromPopulationFile(trees_file);
    dbg("restore() "+n_restores+" from tick "+g_restore+", returning "+
        "to state at save() "+n_saves);

    /*
     * The tick counter community.tick has now been reset to the
     * value it had when save() was called. There are two issues relating
     * to event scheduling which must now be dealt with.
     *
     * 1. There may be additional late events for the tick in which
     * restore() was called, and they are still scheduled to run.
     * So we deactivate all script blocks in the "late" cycle to avoid
     * unexpected problems. They will be automatically reactivated at the
     * start of the next tick.
     */
    sb = community.allScriptBlocks;
    sb[sb.type == "late"].active = F;

    /*
     * 2. The late events below were run in the save() tick,
     * but after the save() call. We execute these again here, because
     * the next late events to run will be for community.tick + 1.
     * Note that the save() event is indistinguishable from the other
     * late events in this tick, so we set a flag `restore_function`
     * to signal the save() function not to save again.
     */
    g = community.tick;
    sim.setValue("restore_function", T);
    for (sb in community.allScriptBlocks) {
        if (sb.type == "late" & g >= sb.start & g <= sb.end) {
            self = sb;
            executeLambda(sb.source);
        }
    }
    sim.setValue("restore_function", F);
}



1 early() {
    // save/restore bookkeeping
    sim.setValue("n_restores", 0);
    sim.setValue("n_saves", 0);
    sim.setValue("restore_function", F);

    /*
     * Create initial populations and migration rates.
     */

    // Initial populations.
    for (i in 0:(num_populations-1)) {
        if (N[0,i] > 0) {
            check_size(i, N[0,i], community.tick);
            dbg("p = sim.addSubpop("+i+", "+N[0,i]+");");
            p = sim.addSubpop(i, N[0,i]);
            dbg("p.name = '"+pop_names[i]+"';");
            p.name = pop_names[i];
        }
    }

    if (length(sim.subpopulations) == 0) {
        err("No populations with non-zero size in tick 1.");
    }

    // The end of the burn-in is the starting tick, and corresponds to
    // time T_start. All remaining events are relative to this tick.
    N_max = max(N[0,0:(num_populations-1)]);
    G_start = community.tick + asInteger(round(burn_in * N_max));
    T_start = max(_T);
    G = G_start + gdiff(T_start, _T);
    G_end = max(G);

    /*
     * Register events occurring at time T_start or more recently.
     */

    // Save/restore events. These should come before all other events.
    if (length(drawn_mutations) > 0) {
        n_checkpoints = 0;
        for (i in 0:(ncol(drawn_mutations)-1)) {
            save = drawn_mutations[4,i] == 1;
            if (save) {
                // Saving the state at more than one timepoint can can cause
                // incorrect conditioning in the rejection samples.
                if (n_checkpoints > 0) {
                    err("Attempt to save state at more than one checkpoint");
                }
                n_checkpoints = n_checkpoints + 1;

                // Unconditionally save the state before the mutation is drawn.
                g = G_start + gdiff(T_start, drawn_mutations[0,i]);
                community.registerLateEvent(NULL, "{save();}", g, g);
            }
        }
    }
    if (length(condition_on_allele_frequency) > 0) {
        for (i in 0:(ncol(condition_on_allele_frequency)-1)) {
            g_start = G_start + gdiff(T_start, condition_on_allele_frequency[0,i]);
            g_end = G_start + gdiff(T_start, condition_on_allele_frequency[1,i]);
            mut_type = asInteger(condition_on_allele_frequency[2,i]);
            pop_id = asInteger(condition_on_allele_frequency[3,i]);
            op = op_types[asInteger(drop(condition_on_allele_frequency[4,i]))];
            af = condition_on_allele_frequency[5,i];

            if (g_start > g_end) {
                err("Attempt to register AF conditioning callback with g_start="+
                    g_start+" > g_end="+g_end);
            }

            // Restore state if AF condition not met.
            community.registerLateEvent(NULL,
                "{if (!(af(m"+mut_type+", p"+pop_id+") "+op+" "+af+"))" +
                " restore();}",
                g_start, g_end);
        }
    }

    // Split events.
    if (length(subpopulation_splits) > 0 ) {
        for (i in 0:(ncol(subpopulation_splits)-1)) {
            g = G_start + gdiff(T_start, subpopulation_splits[0,i]);
            newpop = drop(subpopulation_splits[1,i]);
            size = asInteger(subpopulation_splits[2,i] / Q);
            oldpop = subpopulation_splits[3,i];
            check_size(newpop, size, g);
            community.registerLateEvent(NULL,
                "{dbg(self.source); " +
                "p = sim.addSubpopSplit("+newpop+","+size+","+oldpop+"); " +
                "p.name = '"+pop_names[newpop]+"';}",
                g, g);
        }
		
		g = G_start + gdiff(T_start, subpopulation_splits[0,1]);
		community.registerLateEvent(NULL,
                "{dbg(self.source); " +
                "p"+2+".setSubpopulationSize(0);}",
                g, g);
    }

    // Population size changes.
    if (num_epochs > 1) {
        for (i in 1:(num_epochs-1)) {
            g = G[i-1];
            for (j in 0:(num_populations-1)) {
                // Change population size if this hasn't already been taken
                // care of by sim.addSubpop() or sim.addSubpopSplit().
                if (N[i,j] != N[i-1,j] & N[i-1,j] != 0) {
                    check_size(j, N[i,j], g);
                    community.registerLateEvent(NULL,
                        "{dbg(self.source); " +
                        "p"+j+".setSubpopulationSize("+N[i,j]+");}",
                        g, g);
                }

                if (growth_rates[i,j] != 0) {
                    growth_phase_start = g+1;
                    if (i == num_epochs-1) {
                        growth_phase_end = G[i];
                    } else {
                        // We already registered a size change at tick G[i].
                        growth_phase_end = G[i] - 1;
                    }

                    if (growth_phase_start >= growth_phase_end) {
                        // Demographic models could have duplicate epoch times,
                        // which should be fixed.
                        warn("growth_phase_start="+growth_phase_start+
                             " >= growth_phase_end="+growth_phase_end);
                        next;
                    }

                    N_growth_phase_end = pop_size_at(G, j, growth_phase_end);
                    check_size(j, N_growth_phase_end, growth_phase_end);

                    N0 = N[i,j];
                    r = Q * growth_rates[i,j];
                    community.registerLateEvent(NULL,
                        "{" +
                            "dbg(self.source); " +
                            "gx=community.tick-"+g+"; " +
                            "size=asInteger(round("+N0+"*exp("+r+"*gx))); " +
                            "p"+j+".setSubpopulationSize(size);" +
                        "}",
                        growth_phase_start, growth_phase_end);
                }
            }
        }

        // Migration rates.
        for (i in 1:(num_epochs-1)) {
            for (j in 0:(num_populations-1)) {
                for (k in 0:(num_populations-1)) {
                    if (j==k | N[i,j] == 0 | N[i,k] == 0) {
                        next;
                    }

                    m_last = Q * migration_matrices[k,j,i-1];
                    m = Q * migration_matrices[k,j,i];
                    if (m == m_last) {
                        // Do nothing if the migration rate hasn't changed.
                        next;
                    }
                    g = G[i-1];
                    community.registerLateEvent(NULL,
                        "{dbg(self.source); " +
                        "p"+j+".setMigrationRates("+k+", "+m+");}",
                        g, g);
                }
            }
        }
    }

    // Admixture pulses.
    if (length(admixture_pulses) > 0 ) {
        for (i in 0:(ncol(admixture_pulses)-1)) {
            g = G_start + gdiff(T_start, admixture_pulses[0,i]);
            dest = asInteger(admixture_pulses[1,i]);
            src = asInteger(admixture_pulses[2,i]);
            rate = admixture_pulses[3,i];
            community.registerLateEvent(NULL,
                "{dbg(self.source); " +
                "p"+dest+".setMigrationRates("+src+", "+rate+");}",
                g, g);
            community.registerLateEvent(NULL,
                "{dbg(self.source); " +
                "p"+dest+".setMigrationRates("+src+", 0);}",
                g+1, g+1);
        }
    }

    // Draw mutations.
    if (length(drawn_mutations) > 0) {
        for (i in 0:(ncol(drawn_mutations)-1)) {
            g = G_start + gdiff(T_start, drawn_mutations[0,i]);
            mut_type = asInteger(drawn_mutations[1,i]);
            pop_id = asInteger(drawn_mutations[2,i]);
            coordinate = asInteger(drawn_mutations[3,i]);
            community.registerLateEvent(NULL,
                "{dbg(self.source); " +
                "add_mut(m"+mut_type+", p"+pop_id+", "+coordinate+");}",
                g, g);
        }
    }

    // Setup fitness callbacks.
    if (length(fitness_callbacks) > 0) {
        for (i in 0:(ncol(fitness_callbacks)-1)) {
            g_start = G_start + gdiff(T_start, fitness_callbacks[0,i]);
            g_end = G_start + gdiff(T_start, fitness_callbacks[1,i]);
            mut_type = asInteger(fitness_callbacks[2,i]);
            pop_id = asInteger(fitness_callbacks[3,i]);
            selection_coeff = Q * fitness_callbacks[4,i];
            dominance_coeff = fitness_callbacks[5,i];

            if (g_start > g_end) {
                err("Attempt to register fitness callback with g_start="+
                    g_start+" > g_end="+g_end);
            }

            /* We explicitly format() here to prevent integral-valued floats
             * from getting converted to integers during string interpolation
             * (this triggers a type error when the fitness callback runs). */
            f_hom = format("%e", 1 + selection_coeff);
            f_het = format("%e", 1 + selection_coeff * dominance_coeff);

            /* "All populations" is encoded by a negative value of pop_id. */
            if (pop_id < 0) {
                community.registerLateEvent(NULL,
                    "{dbg('s="+selection_coeff+", h="+dominance_coeff+
                    " for m"+mut_type+" globally');}",
                    g_start, g_start);
                community.registerLateEvent(NULL,
                    "{dbg('s, h defaults for m"+mut_type+" globally');}",
                    g_end, g_end);
                sim.registerMutationEffectCallback(NULL,
                    "{if (homozygous) return "+f_hom+"; else return "+f_het+";}",
                    mut_type, NULL, g_start, g_end);
            } else {
                community.registerLateEvent(NULL,
                    "{dbg('s="+selection_coeff+", h="+dominance_coeff+
                    " for m"+mut_type+" in p"+pop_id+"');}",
                    g_start, g_start);
                community.registerLateEvent(NULL,
                    "{dbg('s, h defaults for m"+mut_type+" in p"+pop_id+"');}",
                    g_end, g_end);
                sim.registerMutationEffectCallback(NULL,
                    "{if (homozygous) return "+f_hom+"; else return "+f_het+";}",
                    mut_type, pop_id, g_start, g_end);
            }
        }
    }

    // Sample individuals.
 	pop1 = drop(sampling_episodes[0,0]);
	n1 = sampling_episodes[1,0];
	pop2 = drop(sampling_episodes[0,1]);
	n2 = sampling_episodes[1,1];
	g = G_start + gdiff(T_start, sampling_episodes[2,0]);
	pop1 = drop(sampling_episodes[0,0]);
	community.registerLateEvent(NULL,
		"{dbg(self.source); " +
		"ind"+pop1+"=p"+pop1+".sampleIndividuals("+n1+").genomes;" +
		"ind"+pop2+"=p"+pop2+".sampleIndividuals("+n2+").genomes;" +
		"c(ind0,ind1).outputVCF(filePath=\"test.vcf\""+", simplifyNucleotides=T);}",
		g, g);					
	
    if (G_start > community.tick) {
        dbg("Starting burn-in...");
    }

    if (dry_run) {
        sim.simulationFinished();
    }
}



///
/// Debugging output
///

// Print out selection coefficients for every new mutation:
// this is for development purposes, and the format of this output
// is subject to change or may even be removed!
// Header:
1 late() {
    if (verbosity >= 3) {
        dbg(paste(c("dbg_selection_coeff:",
                    "selectionCoeff",
                    "id",
                    "position"),
                  sep="	"));
    }
}

// Content:
1: late() {
    // Print out selection coefficients for every new mutation:
    // this is for development purposes, and the format of this output
    // is subject to change or may even be removed!
    if (verbosity >= 3) {
        new = (sim.mutations.originTick == community.tick);
        for (mut in sim.mutations[new]) {
            dbg(paste(c("dbg_selection_coeff:",
                        mut.selectionCoeff,
                        mut.id,
                        mut.position),
                      sep="	"));
        }
    }
}

// Save genomic element type information in tree sequence metadata
// This is for development purposes, and the format of this metadata
// is subject to change or may even be removed!
1 early() {
    if (verbosity >= 3) {
        // recombination map
        metadata.setValue(
            "recombination_rates",
            sim.chromosome.recombinationRates
        );
        metadata.setValue(
            "recombination_ends",
            sim.chromosome.recombinationEndPositions
        );
        // mutationTypes
        muts = Dictionary();
        for (mt in sim.mutationTypes) {
            mut_info = Dictionary(
                "distributionParams", mt.distributionParams,
                "distributionType", mt.distributionType,
                "dominanceCoeff", mt.dominanceCoeff
            );
            muts.setValue(asString(mt.id), mut_info);
        }
        metadata.setValue("mutationTypes", muts);
        // genomicElementTypes
        ge_starts = sim.chromosome.genomicElements.startPosition;
        ge_ends= sim.chromosome.genomicElements.endPosition;
        ge_types= sim.chromosome.genomicElements.genomicElementType.id;
        ges = Dictionary();
        for (gt in sim.genomicElementTypes) {
            gt_info = Dictionary(
                "mutationTypes", gt.mutationTypes.id,
                "mutationFractions", gt.mutationFractions,
                "intervalStarts", ge_starts[ge_types == gt.id],
                "intervalEnds", ge_ends[ge_types == gt.id]
            );
            ges.setValue(asString(gt.id), gt_info);
        }
        metadata.setValue("genomicElementTypes", ges);
        // mutation rates
        mr = Dictionary(
            "rates", sim.chromosome.mutationRates,
            "ends", sim.chromosome.mutationEndPositions
        );
        metadata.setValue("mutationRates", mr);
    }
}
