"""The fleet: run many experiments on the hosts std-00..03 (standard library only).

The Mac makes a git snapshot of the working tree and pushes it to every host
(``snapshot``, ``provision``). Jobs wait in a queue of small files on the NAS
(``jobs``, ``queue``); one worker per host claims them by ``os.rename``,
starts each run as a process pinned to its slot (``slots``, ``worker``),
classifies how it ended by the run contract (``contract``), and moves the
finished run to the NAS. ``cli`` is ``just fleet <cmd>``; ``fake`` is a stand-in
program that honors the run contract, for the tests and for smoke runs.

Nothing here imports torch or hydra, nor the other modules of ``exp.lib`` but
``exp.lib.runs`` (stdlib only, for ``fleet table``, imported when asked): the
CLI runs under the plain Python of the Mac.
"""
