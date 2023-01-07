Md Zahid Khan
Practical Assignment 1
Q – 1 Write a JAVA Program to implement built-in support (java.util.Observable) Weather station with 
members temperature, humidity, pressure and methods mesurmentsChanged(), setMesurment(), 
getTemperature(), getHumidity(), getPressure().
Ans – Step – 1 Observer.java
package weatherProject;
public interface Observer {
public void update(float temp, float humidity, float pressure);
}
Step – 2 DisplayElement.java
package weatherProject;
public interface DisplayElement {
public void display();
}
Step – 3 Subject.java
package weatherProject;
public interface Subject {
public void registerObserver(Observer o);
public void removeObserver(Observer o);
public void notifyObservers();
}
Step – 4 WetaherData.java
package weatherProject;
import java.util.*;
public class WeatherData implements Subject {
private ArrayList<Observer> observers;
private float temperature;
private float humidity;
private float pressure;
public WeatherData() {
 observers = new ArrayList<>();
}
@Override
public void registerObserver(Observer o) {
 observers.add(o);
}
@Override
public void removeObserver(Observer o) {
 int i = observers.indexOf(o);
 if (i >= 0) {
 observers.remove(i);
} }
@Override
public void notifyObservers() {
 for (int i = 0; i < observers.size(); i++) {
 Observer observer = (Observer)observers.get(i);
 observer.update(temperature, humidity, pressure);
 } }
public void measurementsChanged() {
 notifyObservers();
}
public void setMeasurements(float temperature, float humidity, float
pressure) {
 this.temperature = temperature;
 this.humidity = humidity;
 this.pressure = pressure;
 measurementsChanged();
}
public float getTemperature() {
 return temperature;
}
public float getHumidity() {
 return humidity;
}
public float getPressure() {
 return pressure;
} }
Step – 5 StaticDisplay.java
package weatherProject;
public class StatisticsDisplay implements Observer, DisplayElement {
private float maxTemp = 0.0f;
private float minTemp = 200;
private float tempSum= 0.0f;
private int numReadings;
private WeatherData weatherData;
public StatisticsDisplay(WeatherData weatherData) {
 this.weatherData = weatherData;
 weatherData.registerObserver(this);
}
@Override
public void update(float temp, float humidity, float pressure) {
 tempSum += temp;
 numReadings++;
 if (temp > maxTemp) {
 maxTemp = temp;
 }
 if (temp < minTemp) {
 minTemp = temp;
 }
 display();
}
@Override
public void display() {
 System.out.println("Avg/Max/Min temperature = " + (tempSum / 
numReadings)
 + "/" + maxTemp + "/" + minTemp);
} }
Step – 6 WeatherStation.java
package weatherProject;
public class WeatherStation {
public static void main(String[] args) {
 WeatherData weatherData = new WeatherData();
 
 CurrentConditionsDisplay currentDisplay = 
 new CurrentConditionsDisplay(weatherData);
 StatisticsDisplay statisticsDisplay = new
StatisticsDisplay(weatherData);
 ForecastDisplay forecastDisplay = new ForecastDisplay(weatherData);
 weatherData.setMeasurements(80, 65, 30.4f);
 weatherData.setMeasurements(82, 70, 29.2f);
 weatherData.setMeasurements(78, 90, 29.2f);
} }
Step – 7 HeatIndexDisplay.java
package weatherProject;
public class HeatIndexDisplay implements Observer, DisplayElement {
float heatIndex = 0.0f;
private WeatherData weatherData;
public HeatIndexDisplay(WeatherData weatherData) {
 this.weatherData = weatherData;
 weatherData.registerObserver(this);
}
@Override
public void update(float t, float rh, float pressure) {
 heatIndex = computeHeatIndex(t, rh);
 display();
}
private float computeHeatIndex(float t, float rh) {
 float index = (float)((16.923 + (0.185212 * t) + (5.37941 * rh) -
(0.100254 * t * rh) 
 + (0.00941695 * (t * t)) + (0.00728898 * (rh * rh)) 
 + (0.000345372 * (t * t * rh)) - (0.000814971 * (t * rh * rh)) +
 (0.0000102102 * (t * t * rh * rh)) - (0.000038646 * (t * t * t)) + 
(0.0000291583 * 
 (rh * rh * rh)) + (0.00000142721 * (t * t * t * rh)) + 
 (0.000000197483 * (t * rh * rh * rh)) - (0.0000000218429 * (t * t * 
t * rh * rh)) +
 0.000000000843296 * (t * t * rh * rh * rh)) -
 (0.0000000000481975 * (t * t * t * rh * rh * rh)));
 return index;
}
public void display() {
 System.out.println("Heat index is " + heatIndex);
} }
Step – 8 ForecastDisplay.java
package weatherProject;
public class ForecastDisplay implements Observer, DisplayElement {
private float currentPressure = 29.92f; 
private float lastPressure;
private WeatherData weatherData;
public ForecastDisplay(WeatherData weatherData) {
 this.weatherData = weatherData;
 weatherData.registerObserver(this);
}
@Override
public void update(float temp, float humidity, float pressure) {
 lastPressure = currentPressure;
 currentPressure = pressure;
 display();
}
public void display() {
 System.out.print("Forecast: ");
 if (currentPressure > lastPressure) {
 System.out.println("Improving weather on the way!");
 } else if (currentPressure == lastPressure) {
 System.out.println("More of the same");
 } else if (currentPressure < lastPressure) {
 System.out.println("Watch out for cooler, rainy weather");
 } } }
Step – 9 WeatherStation.java
package weatherProject;
public class WeatherStation {
public static void main(String[] args) {
 WeatherData weatherData = new WeatherData();
 
 CurrentConditionsDisplay currentDisplay = 
 new CurrentConditionsDisplay(weatherData);
 StatisticsDisplay statisticsDisplay = new
StatisticsDisplay(weatherData);
 ForecastDisplay forecastDisplay = new ForecastDisplay(weatherData);
 weatherData.setMeasurements(80, 65, 30.4f);
 weatherData.setMeasurements(82, 70, 29.2f);
 weatherData.setMeasurements(78, 90, 29.2f);
} }
Q 2 - Write a Java Program to implement I/O Decorator for converting uppercase letters to lower case letters.
Ans –
import java.io.*;
class LowerCaseInputStream extends FilterInputStream
{
public LowerCaseInputStream(InputStream in)
{
super(in);
}
public int read() throws IOException
{
int c = super.read();
return (c == -1 ? c : Character.toLowerCase((char)c));
}
public int read(byte[] b, int offset, int len) throws IOException
{
int result = super.read(b, offset, len);
for (int i = offset; i>offset+result; i++)
{
b[i] = (byte)Character.toLowerCase((char)b[i]);
}
return result;
} }
public class InputTest
{
public static void main(String[] args) throws IOException
{
int c;
try
{
InputStream in =new LowerCaseInputStream(new 
BufferedInputStream(new
FileInputStream("test.txt")));
while((c = in.read()) >= 0)
{
System.out.print((char)c);
}
in.close();
}
catch (IOException e)
{
e.printStackTrace();
} } }
Q – 3 Write a Java Program to implement Factory method for Pizza Store with createPizza(), orederPizza(), 
prepare(), Bake(), cut(), box(). Use this to create variety of pizza’s like NyStyleCheesePizza, 
ChicagoStyleCheesePizza etc.
Ans –
import java.util.ArrayList;
abstract class Pizza {
 String name;
 String dough;
 String sauce;
 ArrayList toppings = new ArrayList();
 public String getName() {
 return name;
 }
 public void prepare() {
 System.out.println("Preparing " + name);
 }
 public void bake() {
 System.out.println("Baking " + name);
 }
 public void cut() {
 System.out.println("Cutting " + name);
 }
 public void box() {
 System.out.println("Boxing " + name);
 }
 public String toString() {
 // code to display pizza name and ingredients
 StringBuffer display = new StringBuffer();
 display.append("---- " + name + " ----\n");
 display.append(dough + "\n");
 display.append(sauce + "\n");
 for (int i = 0; i < toppings.size(); i++) {
 display.append((String) toppings.get(i) + "\n");
 }
 return display.toString();
 } }
class CheesePizza extends Pizza {
 public CheesePizza() {
 name = "Cheese Pizza";
 dough = "Regular Crust";
 sauce = "Marinara Pizza Sauce";
 toppings.add("Fresh Mozzarella");
 toppings.add("Parmesan");
 } }
class ClamPizza extends Pizza {
 public ClamPizza() {
 name = "Clam Pizza";
 dough = "Thin crust";
 sauce = "White garlic sauce";
 toppings.add("Clams");
 toppings.add("Grated parmesan cheese");
 } }
class VeggiePizza extends Pizza {
 public VeggiePizza() {
 name = "Veggie Pizza";
 dough = "Crust";
 sauce = "Marinara sauce";
 toppings.add("Shredded mozzarella");
 toppings.add("Grated parmesan");
 toppings.add("Diced onion");
 toppings.add("Sliced mushrooms");
 toppings.add("Sliced red pepper");
 toppings.add("Sliced black olives");
 } }
class PepperoniPizza extends Pizza {
 public PepperoniPizza() {
 name = "Pepperoni Pizza";
 dough = "Crust";
 sauce = "Marinara sauce";
 toppings.add("Sliced Pepperoni");
 toppings.add("Sliced Onion");
 toppings.add("Grated parmesan cheese");
 } }
class SimplePizzaFactory {
 public Pizza createPizza(String type) {
 Pizza pizza = null;
 if (type.equals("cheese")) {
 pizza = new CheesePizza();
 } else if (type.equals("pepperoni")) {
 pizza = new PepperoniPizza();
 } else if (type.equals("clam")) {
 pizza = new ClamPizza();
 } else if (type.equals("veggie")) {
 pizza = new VeggiePizza();
 }
 return pizza;
 }
}
class PizzaStore {
 SimplePizzaFactory factory;
 public PizzaStore(SimplePizzaFactory factory) {
 this.factory = factory;
 }
 public Pizza orderPizza(String type) {
 Pizza pizza;
 pizza = factory.createPizza(type);
 pizza.prepare();
 pizza.bake();
 pizza.cut();
 pizza.box();
 return pizza;
 } }
public class PizzaTestDrive {
 public static void main(String[] args) {
 SimplePizzaFactory factory = new SimplePizzaFactory();
 PizzaStore store = new PizzaStore(factory);
 Pizza pizza = store.orderPizza("cheese");
 System.out.println("We ordered a " + pizza.getName() + "\n");
 pizza = store.orderPizza("veggie");
 System.out.println("We ordered a " + pizza.getName() + "\n");
 } }
Q – 4 - Write a Java Program to implement Singleton pattern for multithreading.
Ans –
public class Singleton_Mutlthreading {
 private static volatile Singleton_Mutlthreading instance;
 private static Object mutex = new Object();
 private Singleton_Mutlthreading() {
 }
 public static Singleton_Mutlthreading getInstance() {
 Singleton_Mutlthreading result = instance;
 if (result == null) {
 synchronized (mutex) {
 result = instance;
 if (result == null)
 instance = result = new Singleton_Mutlthreading();
 }
 }
 return result;
 }
 public static void main(String args[]) {
 // Singleton_Mutlthreading s=new Singleton_Mutlthreading();
 Singleton_Mutlthreading s = Singleton_Mutlthreading.getInstance();
 System.out.println(s);
 Singleton_Mutlthreading s1 = Singleton_Mutlthreading.getInstance();
 System.out.println(s1);
 } }
Q – 5 Write a Java Program to implement command pattern to test Remote Control.
Ans –
// An interface for command
interface Command {
 public void execute();
}
// Light class and its corresponding command
// classes
class Light {
 public void on() {
 System.out.println("Light is on");
 }
 public void off() {
 System.out.println("Light is off");
 } }
class LightOnCommand implements Command {
 Light light;
 // The constructor is passed the light it
 // is going to control.
 public LightOnCommand(Light light) {
 this.light = light;
 }
 public void execute() {
 light.on();
 } }
class LightOffCommand implements Command {
 Light light;
 public LightOffCommand(Light light) {
 this.light = light;
 }
 public void execute() {
 light.off();
 } }
// Stereo and its command classes
class Stereo {
 public void on() {
 System.out.println("Stereo is on");
 }
 public void off() {
 System.out.println("Stereo is off");
 }
 public void setCD() {
 System.out.println("Stereo is set " +
 "for CD input");
 }
 public void setDVD() {
 System.out.println("Stereo is set" +
 " for DVD input");
 }
 public void setRadio() {
 System.out.println("Stereo is set" +
 " for Radio");
 }
 public void setVolume(int volume) {
 // code to set the volume
 System.out.println("Stereo volume set"
 + " to " + volume);
 } }
class StereoOffCommand implements Command {
 Stereo stereo;
 public StereoOffCommand(Stereo stereo) {
 this.stereo = stereo;
 }
 public void execute() {
 stereo.off();
 } }
class StereoOnWithCDCommand implements Command {
 Stereo stereo;
 public StereoOnWithCDCommand(Stereo stereo) {
 this.stereo = stereo;
 }
 public void execute() {
 stereo.on();
 stereo.setCD();
 stereo.setVolume(11);
 } }
// A Simple remote control with one button
class SimpleRemoteControl {
 Command slot; // only one button
 public SimpleRemoteControl() {
 }
 public void setCommand(Command command) {
 // set the command the remote will
 // execute
 slot = command;
 }
 public void buttonWasPressed() {
 slot.execute();
 } }
// Driver class
class RemoteControlTest1 {
 public static void main(String[] args) {
 SimpleRemoteControl remote = new SimpleRemoteControl();
 Light light = new Light();
 Stereo stereo = new Stereo();
 // we can change command dynamically
 remote.setCommand(new LightOnCommand(light));
 remote.buttonWasPressed();
 remote.setCommand(new StereoOnWithCDCommand(stereo));
 remote.buttonWasPressed();
 remote.setCommand(new StereoOffCommand(stereo));
 remote.buttonWasPressed();
 } }

Md Zahid Khan
Practical Assignment 2
6. Write a Java Program to implement undo command to test Ceiling fan. 
Program:
public class CeilingFan {
public static final int HIGH = 3; 
public static final int MEDIUM = 2; 
public static final int LOW = 1; 
public static final int OFF = 0;
String location; 
int speed;
public CeilingFan(String location) { 
this.location = location;
speed = OFF; }
public void high() { 
speed = HIGH;
// code to set fan to high
}
public void medium() { 
speed = MEDIUM;
// code to set fan to medium
}
public void low() { 
speed = LOW;
// code to set fan to low
}
public void off() { 
speed = OFF;
// code to turn fan off
}
public int getSpeed() { 
return speed; } }
public class CeilingFanHighCommand implements Command { 
CeilingFan ceilingFan;
int prevSpeed;
public CeilingFanHighCommand(CeilingFan ceilingFan) { 
this.ceilingFan = ceilingFan; }
public void execute() {
prevSpeed = ceilingFan.getSpeed(); 
ceilingFan.high();
}
public void undo() {
if (prevSpeed == CeilingFan.HIGH) { 
ceilingFan.high();
} else if (prevSpeed == CeilingFan.MEDIUM) { 
ceilingFan.medium();
} else if (prevSpeed == CeilingFan.LOW) { 
ceilingFan.low();
} else if (prevSpeed == CeilingFan.OFF) { 
ceilingFan.off();
} } }
public class CeilingFanMediumCommand implements Command { 
CeilingFan ceilingFan;
int prevSpeed;
public CeilingFanMediumCommand(CeilingFan ceilingFan) { 
this.ceilingFan = ceilingFan; }
public void execute() {
prevSpeed = ceilingFan.getSpeed(); 
ceilingFan.medium(); }
public void undo() { if (prevSpeed == CeilingFan.HIGH) { 
ceilingFan.high();
} else if (prevSpeed == CeilingFan.MEDIUM) { 
ceilingFan.medium();
} else if (prevSpeed == CeilingFan.LOW) { 
ceilingFan.low();
} else if (prevSpeed == CeilingFan.OFF) { 
ceilingFan.off();
} } }
public class CeilingFanOffCommand implements Command { 
CeilingFan ceilingFan;
int prevSpeed;
public CeilingFanOffCommand(CeilingFan ceilingFan) { 
this.ceilingFan = ceilingFan; }
public void execute() {
prevSpeed = ceilingFan.getSpeed(); 
ceilingFan.off();
}
public void undo() { if (prevSpeed == CeilingFan.HIGH) { 
ceilingFan.high();
} else if (prevSpeed == CeilingFan.MEDIUM) { 
ceilingFan.medium();
} else if (prevSpeed == CeilingFan.LOW) { 
ceilingFan.low();
} else if (prevSpeed == CeilingFan.OFF) { 
ceilingFan.off();
} } }
public interface Command { 
public voidexecute(); 
public void undo();
}
public class RemoteControlWithUndo { 
Command[] onCommands;
Command[] offCommands;
Command undoCommand;
public RemoteControlWithUndo() { 
onCommands = new Command[7]; 
offCommands = new Command[7];
/*Command noCommand = new NoCommand();
for (int i = 0; i < 7; i++) { 
onCommands[i] = noCommand; 
offCommands[i] = noCommand; }
undoCommand = noCommand;*/
}
public void setCommand(int slot, Command onCommand, Command offCommand) { 
onCommands[slot] = onCommand;
offCommands[slot] = offCommand;
}
public void onButtonWasPushed(int slot) { 
onCommands[slot].execute(); 
undoCommand = onCommands[slot];
}
public void offButtonWasPushed(int slot) { 
offCommands[slot].execute(); 
undoCommand = offCommands[slot];
}
public void undoButtonWasPushed() { 
undoCommand.undo();
}
//public String toString() {}
}
public class RemoteLoader {
public static void main(String[] args) {
RemoteControlWithUndo remoteControl = new RemoteControlWithUndo(); 
CeilingFan ceilingFan = new CeilingFan("Living Room");
CeilingFanMediumCommand ceilingFanMedium = new 
CeilingFanMediumCommand(ceilingFan);
CeilingFanHighCommand ceilingFanHigh = new CeilingFanHighCommand(ceilingFan); 
CeilingFanOffCommand ceilingFanOff = new CeilingFanOffCommand(ceilingFan);
remoteControl.setCommand(0, ceilingFanMedium, ceilingFanOff); 
remoteControl.setCommand(1, ceilingFanHigh, ceilingFanOff);
remoteControl.onButtonWasPushed(0); 
remoteControl.offButtonWasPushed(0); 
System.out.println(remoteControl); 
remoteControl.undoButtonWasPushed();
remoteControl.onButtonWasPushed(1); 
System.out.println(remoteControl); 
remoteControl.undoButtonWasPushed();
}}
Output:
7. Write a Java Program to implement Adapter pattern for Enumeration iterator. 
Program:
import java.util.*; 
import java.io.*;
import java.util.Enumeration;
class EnumerationIterator implements Iterator { 
Enumeration enu;
public EnumerationIterator(Enumeration enu) { 
this.enu = enu; }
public boolean hasNext() {
return enu.hasMoreElements();
}
public Object next() {
return enu.nextElement();
}
public void remove() {
throw new UnsupportedOperationException();
} }
class EnumerationDemo {
public static void main(String args[]) { 
Enumeration enu;
Vector<String> vector = new Vector<String>();
vector.add("Shubham");
vector.add("Snehankit"); 
vector.add("Shreyas");
vector.add("Smita"); 
vector.add("Atul");
EnumerationIterator e = new EnumerationIterator(vector.elements());
// Enumeration<String> enumeration = vector.elements();
// Iterator<String> itr=vector.iterator();
/*
* while(itr.hasNext())
* {
* System.out.println(itr.next());
* }
*/
while (e.hasNext()) { 
System.out.println(e.next()); } } }
Output:
8. Write a Java Program to implement Iterator Pattern for Designing Menu like 
Breakfast, Lunch or Dinner Menu.
Program:
import java.util.*;
class MenuItem { 
String name;
String description; 
boolean vegetarian; 
double price;
public MenuItem(String name, String description, boolean vegetarian, 
double price) {
this.name = name; 
this.description = description; 
this.vegetarian = vegetarian; 
this.price = price; }
public String getName() { 
return name; }
public String getDescription() { 
return description; }
public double getPrice()
{
return price;
}
public boolean isVegetarian()
{
return vegetarian; } }
interface Iterator
{
boolean hasNext();
Object next();
}
class MenuIterator implements Iterator
{
MenuItem[] items;
int position = 0;
public MenuIterator(MenuItem[] items) {
this.items = items; }
public Object next()
{
MenuItem menuItem = items[position];
position = position + 1;
return menuItem;
}
public boolean hasNext()
{ if (position >= items.length || items[position] == null) {
return false;
}
else
{
return true;
} }
}
class DinerMenu
{
static final int MAX_ITEMS = 6;
int numberOfItems = 0;
MenuItem[] menuItems;
public DinerMenu()
{
menuItems = new MenuItem[MAX_ITEMS];
addItem("Vegetarian BLT", "(Fakin’) Bacon with lettuce & tomato on
whole wheat", true, 2.99);
addItem("BLT", "Bacon with lettuce & tomato on whole wheat", false,
2.99);
addItem("Soup of the day", "Soup of the day, with a side of potato 
salad", false, 3.29);
addItem("Hotdog", "A hot dog, with saurkraut, relish, onions, topped 
with cheese", false, 3.05);
}
public void addItem(String name, String description, boolean vegetarian, 
double price)
{
MenuItem menuItem = new MenuItem(name, description, vegetarian,
price);
if (numberOfItems >= MAX_ITEMS) { System.err.println("Sorry,menu is full! Can’t add item to menu");
}
else
{ menuItems[numberOfItems] = menuItem; numberOfItems = numberOfItems + 1; } }
/*
* public
* MenuItem[] getMenuItems()
**
* {
*** return
* menuItems;
**
* }
*/
public Iterator createIterator()
{
return new MenuIterator(menuItems);
} }
class PancakeHouseMenu
{
static final int MAX_ITEMS = 6;
int numberOfItems = 0;
MenuItem[] menuItems;
public PancakeHouseMenu()
{
menuItems = new MenuItem[MAX_ITEMS];
addItem("K&B’s Pancake Breakfast", "Pancakes with scrambled eggs, and 
toast", true, 2.99);
addItem("Regular Pancake Breakfast", "Pancakes with fried eggs, 
sausage", false, 2.99);
addItem("Blueberry Pancakes", "Pancakes made with fresh blueberries", 
true, 3.49);
addItem("Waffles", "Waffles, with your choice of blueberries or 
strawberries", true, 3.59);
}
public void addItem(String name, String description, boolean vegetarian, 
double price)
{
MenuItem menuItem = new MenuItem(name, description, vegetarian,
price);
if (numberOfItems >= MAX_ITEMS) {
System.err.println("Sorry, menu is full! Can’t add item to menu");
}
else
{ menuItems[numberOfItems] = menuItem; numberOfItems = numberOfItems + 1; } }
public Iterator createIterator()
{
return new MenuIterator(menuItems);
}
}
class Waitress
{
PancakeHouseMenu pancakeHouseMenu;
DinerMenu dinerMenu;
public Waitress(PancakeHouseMenu pancakeHouseMenu, DinerMenu dinerMenu) {
this.pancakeHouseMenu = pancakeHouseMenu;
this.dinerMenu = dinerMenu; }
public void printMenu()
{
Iterator pancakeIterator = pancakeHouseMenu.createIterator();
Iterator dinerIterator = dinerMenu.createIterator();
System.out.println("MENU\n ---- \nBREAKFAST");
printMenu(pancakeIterator);
System.out.println("\nLUNCH");
printMenu(dinerIterator);
}
private void printMenu(Iterator iterator)
{
while (iterator.hasNext())
{ MenuItem menuItem = (MenuItem) iterator.next();
System.out.print(menuItem.getName() + ", ");
System.out.print(menuItem.getPrice() + " -- ");
System.out.println(menuItem.getDescription());
} } }
public class MenuTestDrive
{
public static void main(String args[])
{
PancakeHouseMenu pancakeHouseMenu = new PancakeHouseMenu();
DinerMenu dinerMenu = new DinerMenu();
Waitress waitress = new Waitress(pancakeHouseMenu, dinerMenu);
waitress.printMenu();
} }
Output:
9. Write a Java Program to implement State Pattern for Gumball Machine. Create 
instance variable that holds current state from there, we just need to handle all
actions,
behaviors and state transition that can happen. For actions we need to implement 
methods to insert a quarter, remove a quarter, turning the crank and display 
gumball. Program:
public class GumballMachine { 
final static int SOLD_OUT = 0;
final static int NO_QUARTER = 1; 
final static int HAS_QUARTER = 2; 
final static int SOLD = 3;
int state = SOLD_OUT; 
int count = 0;
public GumballMachine(int count) { 
this.count = count; if (count > 0) { state = NO_QUARTER; } }
public void insertQuarter() { 
if (state == HAS_QUARTER) { System.out.println("You can’t insert another quarter");
} else if (state == NO_QUARTER) { 
state = HAS_QUARTER; System.out.println("You inserted a quarter");
} else if (state == SOLD_OUT) { System.out.println("You can’t insert a quarter, the machine is
sold out");
} else if (state == SOLD) { System.out.println("Please wait, we’re already giving you a
gumball");
} }
public void ejectQuarter() { 
if (state == HAS_QUARTER) { System.out.println("Quarter returned"); 
state = NO_QUARTER; } else if (state == NO_QUARTER) {
System.out.println("You haven’t inserted a quarter");
} else if (state == SOLD) { System.out.println("Sorry, you already turned the crank");
} else if (state == SOLD_OUT) { System.out.println("You can’t eject, you haven’t inserted a
quarter yet");
} }
public void turnCrank() { 
if (state == SOLD) { System.out.println("Turning twice doesn’t get you another
gumball!");
} else if (state == NO_QUARTER) { System.out.println("You turned but there’s no quarter");
} else if (state == SOLD_OUT) { System.out.println("You turned, but there are no gumballs");
} else if (state == HAS_QUARTER) { 
System.out.println("You turned..."); 
state = SOLD;
dispense();
} }
public void dispense() { 
if (state == SOLD) { System.out.println("A gumball comes rolling out the slot"); 
count = count - 1;
if (count == 0) {
System.out.println("Oops, out of gumballs!"); 
state = SOLD_OUT;
} else {
state = NO_QUARTER; } } else if (state == NO_QUARTER) { 
System.out.println("You need to pay first");
} else if (state == SOLD_OUT) { 
System.out.println("No gumball dispensed");
} else if (state == HAS_QUARTER) { 
System.out.println("No gumball dispensed");
} }
// other methods here like toString() and refill()
}
public class GumballMachineTestDrive { 
public static void main(String[] args) {
GumballMachine gumballMachine = new GumballMachine(5); 
System.out.println(gumballMachine); 
gumballMachine.insertQuarter(); 
gumballMachine.turnCrank(); 
System.out.println(gumballMachine); 
gumballMachine.insertQuarter(); 
gumballMachine.ejectQuarter(); 
gumballMachine.turnCrank(); 
System.out.println(gumballMachine); 
gumballMachine.insertQuarter(); 
gumballMachine.turnCrank(); 
gumballMachine.insertQuarter(); 
gumballMachine.turnCrank(); 
gumballMachine.ejectQuarter(); 
System.out.println(gumballMachine);
gumballMachine.insertQuarter(); 
gumballMachine.insertQuarter(); 
gumballMachine.turnCrank(); 
gumballMachine.insertQuarter(); 
gumballMachine.turnCrank(); 
gumballMachine.insertQuarter(); 
gumballMachine.turnCrank(); 
System.out.println(gumballMachine);
} }
Output:
10. Write a java program to implement Adapter pattern to design Heart Model to 
Beat Model.
Program:
interface Heart_wala { 
public void pump();
public void purify_blood();
}
class Heart_function implements Heart_wala { 
public void pump()
{
System.out.println("Pumps!");
}
public void purify_blood() { 
System.out.println("Purify The Blood!\n");
} }
interface Beat_Wala {
public void oxygen_blood();
}
class Ad_Beat_Wala implements Beat_Wala { 
public void oxygen_blood() {
System.out.println("Oxygen to the Blood!\n");
} }
class Heart_Adapter implements Beat_Wala { 
Heart_wala heart;
public Heart_Adapter(Heart_wala heart) { 
this.heart = heart; }
public void oxygen_blood(){ 
heart.purify_blood();
} }
public class Heart {
public static void main(String args[]) {
Heart_function Heart_function = new Heart_function(); 
Beat_Wala Beat_Wala = new Ad_Beat_Wala();
Beat_Wala Heart_walaAdapter = new Heart_Adapter(Heart_function); 
System.out.println("Heart_functions:");
Heart_function.pump(); 
Heart_function.purify_blood(); 
System.out.println("Beat_Functions"); 
Beat_Wala.oxygen_blood(); 
System.out.println("Important_Work"); 
Heart_walaAdapter.oxygen_blood();
} }
Output: