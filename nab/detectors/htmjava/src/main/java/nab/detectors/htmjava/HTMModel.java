package nab.detectors.htmjava;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import joptsimple.OptionParser;
import joptsimple.OptionSet;
import org.joda.time.DateTimeZone;
import org.numenta.nupic.Connections;
import org.numenta.nupic.Parameters;
import org.numenta.nupic.Parameters.KEY;
import org.numenta.nupic.algorithms.Anomaly;
import org.numenta.nupic.algorithms.SpatialPooler;
import org.numenta.nupic.algorithms.TemporalMemory;
import org.numenta.nupic.network.Layer;
import org.numenta.nupic.network.Network;
import org.numenta.nupic.network.Region;
import org.numenta.nupic.network.sensor.ObservableSensor;
import org.numenta.nupic.network.sensor.Publisher;
import org.numenta.nupic.network.sensor.Sensor;
import org.numenta.nupic.network.sensor.SensorParams;
import org.numenta.nupic.util.Tuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class HTMModel {
    protected static final Logger LOGGER = LoggerFactory.getLogger(HTMModel.class);

    private Network network;

    private Publisher publisher;

    /**
     * Create HTM Model to be used by NAB
     * @param modelParams OPF Model parameters to parameters from
     */
    public HTMModel(JsonNode modelParams) {
        LOGGER.trace("HTMModel({})", modelParams);

        // Create Sensor publisher to push NAB input data to network
        publisher = Publisher.builder()
                .addHeader("timestamp,value")
                .addHeader("datetime,float")
                .addHeader("T,B")
                .build();

        // Get updated model parameters
        Parameters parameters = getModelParameters(modelParams);

        // Create NAB Network
        network = Network.create("NAB Network", parameters)
                .add(Network.createRegion("NAB Region")
                    .add(Network.createLayer("NAB Layer", parameters)
                        .add(Anomaly.create())
                        .add(new TemporalMemory())
                        .add(new SpatialPooler())
                        .add(Sensor.create(ObservableSensor::create,
                                SensorParams.create(SensorParams.Keys::obs, "Manual Input", publisher)))));
    }

    /**
     * Update encoders parameters
     * @param modelParams OPF Model parameters to get encoder parameters from
     * @return Updated Encoder parameters suitable for {@link Parameters.KEY.FIELD_ENCODING_MAP}
     */
    public Map<String, Map<String, Object>> getFieldEncodingMap(JsonNode modelParams) {
        Map<String, Map<String, Object>> fieldEncodings = new HashMap<>();
        String fieldName;
        Map<String, Object> fieldMap;
        JsonNode encoders = modelParams.path("encoders");
        LOGGER.trace("getFieldEncodingMap({})", encoders);
        for (JsonNode node : encoders) {
            if (node.isNull())
                continue;

            fieldName = node.path("fieldname").textValue();
            fieldMap = fieldEncodings.get(fieldName);
            if (fieldMap == null) {
                fieldMap = new HashMap<>();
                fieldMap.put("fieldName", fieldName);
                fieldEncodings.put(fieldName, fieldMap);
            }
            fieldMap.put("encoderType", node.path("type").textValue());
            if (node.has("timeOfDay")) {
                JsonNode timeOfDay = node.get("timeOfDay");
                fieldMap.put("fieldType", "datetime");
                fieldMap.put(KEY.DATEFIELD_PATTERN.getFieldName(), "YYYY-MM-dd HH:mm:ss");
                fieldMap.put(KEY.DATEFIELD_TOFD.getFieldName(),
                        new Tuple(timeOfDay.get(0).asInt(), timeOfDay.get(1).asDouble()));
            } else {
                fieldMap.put("fieldType", "float");
            }
            if (node.has("resolution")) {
                fieldMap.put("resolution", node.get("resolution").asDouble());
            }
        }
        LOGGER.trace("getFieldEncodingMap => {}", fieldEncodings);
        return fieldEncodings;
    }

    /**
     * Update Spatial Pooler parameters
     * @param modelParams OPF Model parameters to get spatial pooler parameters from
     * @return Updated Spatial Pooler parameters
     */
    public Parameters getSpatialPoolerParams(JsonNode modelParams) {
        Parameters p = Parameters.getSpatialDefaultParameters();
        JsonNode spParams = modelParams.path("spParams");
        LOGGER.trace("getSpatialPoolerParams({})", spParams);
        if (spParams.has("spVerbosity")) {
            p.setParameterByKey(KEY.SP_VERBOSITY, spParams.get("spVerbosity").asInt());
        }
        if (spParams.has("columnCount")) {
            p.setParameterByKey(KEY.COLUMN_DIMENSIONS, new int[]{spParams.get("columnCount").asInt()});
        }
        if (spParams.has("maxBoost")) {
            p.setParameterByKey(KEY.MAX_BOOST, spParams.get("maxBoost").asDouble());
        }
        if (spParams.has("synPermInactiveDec")) {
            p.setParameterByKey(KEY.SYN_PERM_INACTIVE_DEC, spParams.get("synPermInactiveDec").asDouble());
        }
        if (spParams.has("synPermConnected")) {
            p.setParameterByKey(KEY.SYN_PERM_CONNECTED, spParams.get("synPermConnected").asDouble());
        }
        if (spParams.has("synPermActiveInc")) {
            p.setParameterByKey(KEY.SYN_PERM_ACTIVE_INC, spParams.get("synPermActiveInc").asDouble());
        }
        if (spParams.has("seed")) {
            p.setParameterByKey(KEY.SEED, spParams.get("seed").asInt());
        }
        if (spParams.has("numActiveColumnsPerInhArea")) {
            p.setParameterByKey(KEY.NUM_ACTIVE_COLUMNS_PER_INH_AREA, spParams.get("numActiveColumnsPerInhArea").asDouble());
        }
        if (spParams.has("globalInhibition")) {
            p.setParameterByKey(KEY.GLOBAL_INHIBITION, spParams.get("globalInhibition").asBoolean());
        }
        if (spParams.has("potentialPct")) {
            p.setParameterByKey(KEY.POTENTIAL_PCT, spParams.get("potentialPct").asDouble());
        }

        LOGGER.trace("getSpatialPoolerParams => {}", p);
        return p;
    }

    /**
     * Update Temporal Memory parameters
     * @param modelParams OPF Model parameters to get Temporal Memory parameters from
     * @return Updated Temporal Memory parameters
     */
    public Parameters getTemporalMemoryParams(JsonNode modelParams) {
        Parameters p = Parameters.getTemporalDefaultParameters();
        JsonNode tpParams = modelParams.path("tpParams");
        LOGGER.trace("getTemporalMemoryParams({})", tpParams);
        if (tpParams.has("columnCount")) {
            p.setParameterByKey(KEY.COLUMN_DIMENSIONS, new int[]{tpParams.get("columnCount").asInt()});
        }
        if (tpParams.has("inputWidth")) {
            p.setParameterByKey(KEY.INPUT_DIMENSIONS, new int[]{tpParams.get("inputWidth").asInt()});
        }
        if (tpParams.has("activationThreshold")) {
            p.setParameterByKey(KEY.ACTIVATION_THRESHOLD, tpParams.get("activationThreshold").asInt());
        }
        if (tpParams.has("cellsPerColumn")) {
            p.setParameterByKey(KEY.CELLS_PER_COLUMN, tpParams.get("cellsPerColumn").asInt());
        }
        if (tpParams.has("permanenceInc")) {
            p.setParameterByKey(KEY.PERMANENCE_INCREMENT, tpParams.get("permanenceInc").asDouble());
        }
        if (tpParams.has("minThreshold")) {
            p.setParameterByKey(KEY.MIN_THRESHOLD, tpParams.get("minThreshold").asInt());
        }
        if (tpParams.has("verbosity")) {
            p.setParameterByKey(KEY.TM_VERBOSITY, tpParams.get("verbosity").asInt());
        // +        "globalDecay": 0.0,
        }
        if (tpParams.has("initialPerm")) {
            p.setParameterByKey(KEY.INITIAL_PERMANENCE, tpParams.get("initialPerm").asDouble());
        // +        "maxAge": 0,
        // +        "maxSegmentsPerCell": 128,
        // +        "maxSynapsesPerSegment": 128,
        }
        if (tpParams.has("permanenceDec")) {
            p.setParameterByKey(KEY.PERMANENCE_DECREMENT, tpParams.get("permanenceDec").asDouble());
        }
        if (tpParams.has("predictedSegmentDecrement")) {
            p.setParameterByKey(KEY.PREDICTED_SEGMENT_DECREMENT, tpParams.get("predictedSegmentDecrement").asDouble());
        }
        if (tpParams.has("seed")) {
            p.setParameterByKey(KEY.SEED, tpParams.get("seed").asInt());
        }
        if (tpParams.has("newSynapseCount")) {
            p.setParameterByKey(KEY.MAX_NEW_SYNAPSE_COUNT, tpParams.get("newSynapseCount").intValue());
        }

        LOGGER.trace("getTemporalMemoryParams => {}", p);
        return p;
    }

    /**
     * Update Sensor parameters
     * @param modelParams OPF Model parameters to get Sensor parameters from
     * @return Updated Sensor parameters
     */
    public Parameters getSensorParams(JsonNode modelParams) {
        JsonNode sensorParams = modelParams.path("sensorParams");
        LOGGER.trace("getSensorParams({})", sensorParams);
        Map<String, Map<String, Object>> fieldEncodings = getFieldEncodingMap(sensorParams);
        Parameters p = Parameters.empty();
        p.setParameterByKey(KEY.CLIP_INPUT, true);
        p.setParameterByKey(KEY.FIELD_ENCODING_MAP, fieldEncodings);

        LOGGER.trace("getSensorParams => {}", p);
        return p;
    }

    /**
     * Update NAB parameters
     * @param params OPF parameters to get NAB model parameters from
     * @return Updated Model parameters
     */
    public Parameters getModelParameters(JsonNode params) {
        JsonNode modelParams = params.path("modelParams");
        LOGGER.trace("getModelParameters({})", modelParams);
        Parameters p = Parameters.getAllDefaultParameters()
                .union(getSpatialPoolerParams(modelParams))
                .union(getTemporalMemoryParams(modelParams))
                .union(getSensorParams(modelParams));

        LOGGER.trace("getModelParameters => {}", p);
        return p;
    }

    public Publisher getPublisher() {
        return publisher;
    }

    public Network getNetwork() {
        return network;
    }


    public void showDebugInfo() {
        Region region = network.getHead();
        Layer<?> layer = region.lookup("NAB Layer");
        Connections connections = layer.getConnections();
        double[] cycles = connections.getActiveDutyCycles();
        int spActive = 0;
        for (int i = 0; i < cycles.length; i++) {
            if (cycles[i] > 0) {
                spActive++;
            }
        }
        LOGGER.debug("SP ActiveDutyCycles: {}", spActive);
    }

    /**
     * Launch htm.java NAB detector
     *
     * Usage:
     *      As a standalone application (for debug purpose only):
     *
     *          java -jar htm.java-nab.jar "{\"modelParams\":{....}}" < nab_data.csv > anomalies.out
     *
     *      For complete list of command line options use:
     *
     *          java -jar htm.java-nab.jar --help
     *
     *      As a NAB detector (see 'htmjava_detector.py'):
     *
     *          python run.py --detect --score --normalize -d htmjava
     *
     *      Logging options, see "log4j.properties":
     *
     *          - "LOGLEVEL": Controls log output (default: "OFF")
     *          - "LOGGER": Either "CONSOLE" or "FILE" (default: "CONSOLE")
     *          - "LOGFILE": Log file destination (default: "htmjava.log")
     *
     *      For example:
     *
     *          java -DLOGLEVEL=TRACE -DLOGGER=FILE -jar htm.java-nab.jar "{\"modelParams\":{....}}" < nab_data.csv > anomalies.out
     *
     */
    public static void main(String[] args) {
        try {
            LOGGER.trace("main({})",  Arrays.asList(args));
            // Parse command line args
            OptionParser parser = new OptionParser();
            parser.nonOptions("OPF parameters object (JSON)");
            parser.acceptsAll(Arrays.asList("p", "params"), "OPF parameters file (JSON).\n(default: first non-option argument)")
                    .withOptionalArg()
                    .ofType(File.class);
            parser.acceptsAll(Arrays.asList("i", "input"), "Input data file (csv).\n(default: stdin)")
                    .withOptionalArg()
                    .ofType(File.class);
            parser.acceptsAll(Arrays.asList("o", "output"), "Output results file (csv).\n(default: stdout)")
                    .withOptionalArg()
                    .ofType(File.class);
            parser.acceptsAll(Arrays.asList("s", "skip"), "Header lines to skip")
                    .withOptionalArg()
                    .ofType(Integer.class)
                    .defaultsTo(0);
            parser.acceptsAll(Arrays.asList("h", "?", "help"), "Help");
            OptionSet options = parser.parse(args);
            if (args.length == 0 || options.has("h")) {
                parser.printHelpOn(System.out);
                return;
            }

            // Get in/out files
            final PrintStream output;
            final InputStream input;
            if (options.has("i")) {
                input = new FileInputStream((File)options.valueOf("i"));
            } else {
                input = System.in;
            }
            if (options.has("o")) {
                output = new PrintStream((File)options.valueOf("o"));
            } else {
                output = System.out;
            }

            // Parse OPF Model Parameters
            JsonNode params;
            ObjectMapper mapper = new ObjectMapper();
            if (options.has("p")) {
                params = mapper.readTree((File)options.valueOf("p"));
            } else if (options.nonOptionArguments().isEmpty()) {
                throw new IllegalArgumentException("Expecting OPF parameters. See 'help' for more information");
            } else {
                params = mapper.readTree((String)options.nonOptionArguments().get(0));
            }

            // Number of header lines to skip
            int skip = (int) options.valueOf("s");

            // Force timezone to UTC
            DateTimeZone.setDefault(DateTimeZone.UTC);

            // Create NAB Network Model
            HTMModel model = new HTMModel(params);
            Network network = model.getNetwork();

            network.observe().subscribe((inference) -> {
                double score = inference.getAnomalyScore();
                int record = inference.getRecordNum();
                LOGGER.trace("record = {}, score = {}", record, score);
                // Output raw anomaly score
                output.println(score);
            }, (error) -> {
                LOGGER.error("Error processing data", error);
            }, () -> {
                LOGGER.trace("Done processing data");
                if(LOGGER.isDebugEnabled()) {
                    model.showDebugInfo();
                }
            });
            network.start();

            // Pipe data to network
            Publisher publisher = model.getPublisher();
            BufferedReader in = new BufferedReader(new InputStreamReader(input));
            String line;
            while ((line = in.readLine()) != null && line.trim().length() > 0) {
                // Skip header lines
                if (skip > 0) {
                    skip--;
                    continue;
                }
                publisher.onNext(line);
            }
            publisher.onComplete();
            in.close();
            LOGGER.trace("Done publishing data");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
