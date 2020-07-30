
# Comparing Agents

Comparing agents is also executed via config. See [conf/c01_ag_cmp/cmp.prototxt](../conf/c01_ag_cmp/cmp.prototxt) for an example config. The following will play Dipnet model vs 6 Mila bots:

```bash
python run.py --adhoc --cfg conf/c01_ag_cmp/cmp.prototxt
```

Configs for all possible agents are located in [conf/common/agents](../conf/common/agents).

Here's the list of the main agents:

  * `random` - random.
  * `mila` - agent released with the Mila paper.
  * `dipnet` - out agent trained on human data. Samples random actions with temperature and nucleus sampling.
  * `cfr1p` - CFR with one ply search on top of DipNet. The best agent so far.
  * `fp1p` - Fictitious Play on top of DipNet.

 You can plug an agent config into eval config like this:

```bash
python run.py --adhoc --cfg conf/c01_ag_cmp/cmp.prototxt \
  I.agent_one=agents/random I.agent_six=agents/dipnet \
  power_one=ITALY
```

This will play random agent (playing Italy) against 6 dipnet agents, and writes the output to `output.json`.

To run a full comparison suite on the cluster see [slurm/compare_agents.sh](../slurm/compare_agents.sh).

Example usage:

```bash
NUM_TRIALS=100 TIME=$(( 60 * 24 )) bash \
  slurm/compare_agents_fast.sh \
  dipnet mila \
  evals_dipnet_vs_mila \
  agent_one.dipment.temperature=0.5
```

Once all (or some) eval jobs are done, run `./bin/get_compare_results.py <output_dir>` to get aggregated scores.
The results are stored in `/checkpoint/<user>/fairdiplomacy/<eval_name>`.