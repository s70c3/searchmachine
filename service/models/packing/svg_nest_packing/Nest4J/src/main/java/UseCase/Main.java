package UseCase;

import com.qunhe.util.nest.Nest;
import com.qunhe.util.nest.data.*;
import com.qunhe.util.nest.util.Config;
import com.qunhe.util.nest.util.SvgUtil;
import com.qunhe.util.nest.util.coor.NestCoor;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.*;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;



public class Main {

    public static void main(String[] args) throws Exception {
        String path = args[0];
        Integer iterations = Integer.valueOf(args[1]);
        Integer rotations = Integer.valueOf(args[2]);
        System.out.println("path " + path);
        System.out.println("iterations " + iterations);
        System.out.println("rotations " + rotations);

        NestCoor wh = Main.readContainerFromFile(path);
        int w = (int) wh.x;
        int h = (int) wh.y;
        NestPath container = Main.createContainer(w, h);
        List<NestPath> list = Main.readShapesFromFile(path, rotations);

        Config config = new Config();
        config.SPACING = 3;
        config.POPULATION_SIZE = 20;
        config.USE_HOLE = false;

        Nest nest = new Nest(container, list, config, iterations);
        List<List<Placement>> appliedPlacement = nest.startNest();
        List<String> strings = SvgUtil.svgGenerator(list, appliedPlacement, w, h);

        Main.saveSvgFile(strings);
    }

    private static NestCoor readContainerFromFile(String path) throws IOException, ParseException {
        JSONParser parser = new JSONParser();
        Object object = parser
                .parse(new FileReader(path));

        //convert Object to JSONObject
        JSONObject jsonObject = (JSONObject)object;
        JSONObject container = (JSONObject) jsonObject.get("container");
        Long w = (Long) container.get("width");
        Long h = (Long) container.get("height");
        NestCoor wh = new NestCoor(w.intValue(), h.intValue());
        return wh;
    }

    private static List<NestPath> readShapesFromFile(String path, Integer rotations) throws IOException, ParseException {
        JSONParser parser = new JSONParser();
        Object object = parser
                .parse(new FileReader(path));

        //convert Object to JSONObject
        JSONObject jsonObject = (JSONObject)object;

        List<NestPath> shapes = new ArrayList<NestPath>();
        JSONArray objs = (JSONArray) jsonObject.get("shapes");
        Iterator iterator = objs.iterator();

        while (iterator.hasNext()) {
            JSONObject obj = (JSONObject) iterator.next();
            NestPath shape = new NestPath();
            Long id = (Long) obj.get("id");
            shape.bid = id.intValue();
            shape.setRotation(rotations);

            JSONArray points = (JSONArray) obj.get("points");
            Iterator pointsIterator = points.iterator();
            while (pointsIterator.hasNext()) {
                JSONObject point = (JSONObject) pointsIterator.next();
                Double x = (Double) point.get("x");
                Double y = (Double) point.get("y");

                shape.add(x, y);
            }
            shapes.add(shape);
        }
        return shapes;
    }

    private static void printShapeAsJson(NestPath shape){
        System.out.println("{\"id\": "+ shape.bid + ",\"points\": [");
        for (Segment coor : shape.get_segments()){
            System.out.print("{\"x\":" + coor.x + ", \"y\": " + coor.y + "}, ");
        }
        System.out.println("\n]},");
    }

    private static NestPath createPolygon(List<NestCoor> coords){
        NestPath poly = new NestPath();
        for (NestCoor pair : coords) {
            poly.add(pair.x, pair.y);
        }
        return poly;
    }

    private static NestPath createContainer(int width, int height) {
        NestPath container = new NestPath();
        container.add(0, 0);
        container.add(0, height);
        container.add(width, height);
        container.add(width, 0);
        return container;
    }

    private static void saveSvgFile(List<String> strings) throws Exception {
        File f = new File("res.svg");
        if (!f.exists()) {
            f.createNewFile();
        }
        Writer writer = new FileWriter(f, false);
        writer.write("<?xml version=\"1.0\" standalone=\"no\"?>\n" +
                "\n" +
                "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\" \n" +
                "\"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n" +
                " \n" +
                "<svg width=\"100%\" height=\"100%\" version=\"1.1\"\n" +
                "xmlns=\"http://www.w3.org/2000/svg\">\n");
        for(String s : strings){
            writer.write(s);
        }
        writer.write("</svg>");
        writer.close();
    }
}
