width_grid = 10     # x
height_grid = 10    # y
width_local = 182   # 232   # cm
height_local= 182   # cm
dist_to_ground = 57 # cm
up_camera_corrcet = 20 #30 # cm
def grid2local(position_grid):
    x_grid = position_grid[1]
    y_grid = height_grid-1 - position_grid[0]
    
    grid_len_width = width_local / width_grid
    grid_len_height = height_local / height_grid
    
    x_local = grid_len_width / 2.0 + x_grid * grid_len_width
    y_local = grid_len_height / 2.0 + y_grid * grid_len_height
    
    return (x_local, y_local)

def local2world(position_local):
    x_local = position_local[0]
    y_local = position_local[1]
    x_world = x_local - width_local/2.0
    y_world = y_local + dist_to_ground + up_camera_corrcet
    # cm to mm (used by vicon)
    x_world *= 10
    y_world *= 10
    return (x_world, y_world)

if __name__ == '__main__':
    pos_grid = (9,9)
    pos_local = grid2local(pos_grid)
    print(pos_local)
    pos_world = local2world(pos_local)
    print(pos_world)