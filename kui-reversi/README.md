# KUI - Reversi

Student support scripts for the Reversi task of KUI course.

Among the files, the following three are for your interest:
  * `player.py` - a template for your player implementation
  * `reversi_creator.py` - runs the game with GUI
  * `headless_reversi_creator.py` - runs the game without GUI, in terminal

## Implementing your player
Start with `player.py` file. Fill in the `select_move()` method with your implementation. The method receives the current board state and shall output the desired position of your next move.

## Playing a game with your player in terminal

The script `headless_reversi_creator.py` accepts one or two parameters where you can specify the players. E.g.

```bash
> python3 headless_reversi_creator.py player
```

plays a game of your player against another instance of your player. It expects a file called `player.py` that contains class `MyPlayer`. 

You can also specify two players:

```bash
> python3 headless_reversi_creator.py player another_player
```
to two different players against each other. Of course, module `another_player.py` must exist and contain class `MyPlayer`.

## Playing a game with your player using a GUI

You can run the interactive version of the game by

```bash
>> python reversi_creator.py player
```
The arguments are the same as for `headless_reversi_creator.py`. In this case, however, they are not directly run against each other; they players are just added to the menu of available players. You can then select the actual players in the GUI. The GUI always offers a player called `interactive`, i.e. you can let play a human against another human, even if you do not have any player implemented yet.
