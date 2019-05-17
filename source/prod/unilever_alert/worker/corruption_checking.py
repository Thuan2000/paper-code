class CorruptionChecker():
    def __init__(self, points):
        self.points = points

    def cordinate_vs_line(self, x, y, points):
        return y + (points[0][1] - points[1][1])*(x - points[0][0])/(points[1][0] - points[0][0]) - points[0][1]

    def line_warning(self, x, y, vx, vy):
        current_point_vs_line = self.cordinate_vs_line(x,y,self.points)
        predict_point_vs_line = self.cordinate_vs_line(x+vx,y+vy,self.points)
        if current_point_vs_line*(current_point_vs_line - predict_point_vs_line) > 0:
            return True # Going toward the fence
        else :
            return False # Going far away from the fence

    def line_violation(self, box):
        ''' input is BB with [t, l, w, h] format
        Check if bouding box is crossing the line '''
        tl_x = box[0]
        tl_y = box[1]
        w = box[2] - box[0]
        h = box[3] - box[1]
        top_left_vs_line = self.cordinate_vs_line(tl_x,tl_y,self.points)
        top_right_vs_line = self.cordinate_vs_line(tl_x+w,tl_y,self.points)
        bot_left_vs_line = self.cordinate_vs_line(tl_x,tl_y+h,self.points)
        bot_right_vs_line = self.cordinate_vs_line(tl_x+w,tl_y+h,self.points)
        centroid = (tl_x + w/2, tl_y+ h/2)
        centroid_vs_line = self.cordinate_vs_line(centroid[0] ,centroid[1],self.points)
        # print(top_left_vs_line,top_right_vs_line,bot_left_vs_line,bot_right_vs_line,centroid_vs_line)
        if (top_left_vs_line*centroid_vs_line < 0) or (top_right_vs_line*centroid_vs_line < 0) \
        or (bot_left_vs_line*centroid_vs_line < 0) or (bot_right_vs_line*centroid_vs_line < 0) :
            return True
        else :
            return False
